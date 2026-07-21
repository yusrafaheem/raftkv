import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from raftkv.cluster import SimulatedCluster
from raftkv.kv.store import Command, SetCommand
from raftkv.raft.log import LogEntry
from raftkv.raft.node import RaftNode
from raftkv.raft.rpc import AppendEntriesArgs, Message
from raftkv.raft.types import Role


def elected_cluster(node_ids=(1, 2, 3), seed=0, **kwargs):
    c = SimulatedCluster(list(node_ids), seed=seed, **kwargs)
    ok = c.run_until(lambda cl: cl.leader() is not None, max_ticks=300)
    assert ok, "setup failed: no leader elected"
    return c


class TestBasicReplication(unittest.TestCase):
    def test_proposed_command_replicates_to_every_follower_and_commits(self):
        c = elected_cluster(seed=10)
        leader = c.leader()
        result = c.propose(Command("client", 1, SetCommand("x", "1")))
        self.assertIsNotNone(result.index)
        ok = c.run_until(lambda cl: cl.is_committed_everywhere(result.index), max_ticks=200)
        self.assertTrue(ok)
        for node_id in c.node_ids:
            self.assertEqual(c.state_machines[node_id].get("x"), "1")
            self.assertEqual(c.nodes[node_id].log.get(result.index).command.op.key, "x")

    def test_multiple_commands_commit_in_proposed_order(self):
        c = elected_cluster(seed=11)
        results = [
            c.propose(Command("client", i, SetCommand("k", str(i)))) for i in range(1, 6)
        ]
        last_index = results[-1].index
        c.run_until(lambda cl: cl.is_committed_everywhere(last_index), max_ticks=300)
        for node_id in c.node_ids:
            applied = [e.command.op.value for e in c.nodes[node_id].log.entries_from(1)]
            self.assertEqual(applied, ["1", "2", "3", "4", "5"])
        # the final state reflects the *last* write, not some interleaving
        for node_id in c.node_ids:
            self.assertEqual(c.state_machines[node_id].get("k"), "5")

    def test_propose_on_a_follower_is_rejected(self):
        c = elected_cluster(seed=12)
        follower = next(n for n in c.node_ids if n != c.leader())
        result = c.propose(Command("client", 1, SetCommand("x", "1")), via=follower)
        self.assertIsNone(result.index)

    def test_commit_index_does_not_advance_without_a_majority(self):
        # 5-node cluster: partition the leader off with only one follower,
        # leaving it short of a majority (2 of 5).
        c = elected_cluster(node_ids=(1, 2, 3, 4, 5), seed=13)
        leader = c.leader()
        follower = next(n for n in c.node_ids if n != leader)
        others = [n for n in c.node_ids if n not in (leader, follower)]
        c.partition([{leader, follower}, set(others)])

        result = c.propose(Command("client", 1, SetCommand("x", "1")), via=leader)
        self.assertIsNotNone(result.index)
        c.run(50)
        self.assertEqual(c.nodes[leader].commit_index, 0)


class TestLogMatchingAndConflictResolution(unittest.TestCase):
    """Direct unit tests of the AppendEntries receiver logic (Raft section
    5.3): a follower must delete conflicting entries and adopt the
    leader's version, and must never accept entries whose prev-entry
    doesn't match its own log."""

    def test_follower_rejects_append_entries_when_prev_log_term_mismatches(self):
        follower = RaftNode(2, [1, 3])
        follower.current_term = 1
        follower.log.append(LogEntry(term=1, index=1, command="a"))
        args = AppendEntriesArgs(
            term=1, leader_id=1, prev_log_index=1, prev_log_term=99, entries=(), leader_commit=0
        )
        reply = follower.step(Message(1, 2, args))[0]
        self.assertFalse(reply.payload.success)
        self.assertEqual(follower.log.last_index, 1)  # untouched

    def test_follower_truncates_conflicting_suffix_and_adopts_leaders_entries(self):
        follower = RaftNode(2, [1, 3])
        follower.current_term = 1
        follower.log.append(LogEntry(term=1, index=1, command="a"))
        follower.log.append(LogEntry(term=1, index=2, command="stale-b"))
        follower.log.append(LogEntry(term=1, index=3, command="stale-c"))

        # Leader's term-2 entry at index 2 conflicts with follower's term-1
        # entry at index 2 -> follower must drop index 2 onward and adopt
        # the leader's version.
        args = AppendEntriesArgs(
            term=2,
            leader_id=1,
            prev_log_index=1,
            prev_log_term=1,
            entries=(LogEntry(term=2, index=2, command="fresh-b"),),
            leader_commit=0,
        )
        reply = follower.step(Message(1, 2, args))[0]
        self.assertTrue(reply.payload.success)
        self.assertEqual(follower.log.last_index, 2)
        self.assertEqual(follower.log.get(2).command, "fresh-b")

    def test_duplicate_append_entries_is_idempotent(self):
        follower = RaftNode(2, [1, 3])
        follower.current_term = 1
        args = AppendEntriesArgs(
            term=1,
            leader_id=1,
            prev_log_index=0,
            prev_log_term=0,
            entries=(LogEntry(term=1, index=1, command="a"),),
            leader_commit=0,
        )
        follower.step(Message(1, 2, args))
        follower.step(Message(1, 2, args))  # exact same RPC delivered twice
        self.assertEqual(follower.log.last_index, 1)
        self.assertEqual(follower.log.get(1).command, "a")

    def test_follower_advances_commit_index_from_leader_commit_but_not_past_its_own_log(self):
        follower = RaftNode(2, [1, 3])
        follower.current_term = 1
        args = AppendEntriesArgs(
            term=1,
            leader_id=1,
            prev_log_index=0,
            prev_log_term=0,
            entries=(LogEntry(term=1, index=1, command="a"),),
            leader_commit=99,  # leader claims more committed than it just sent
        )
        follower.step(Message(1, 2, args))
        self.assertEqual(follower.commit_index, 1)  # clamped to what we actually have


class TestLeaderOnlyCommitsOwnTermEntries(unittest.TestCase):
    """Raft Figure 8: a leader must not commit an entry from a previous
    term just because it's now replicated to a majority -- it can only
    be committed indirectly, by a later entry *from the leader's current
    term* reaching the majority first."""

    def test_leader_does_not_advance_commit_index_past_a_majority_replicated_prior_term_entry_alone(
        self,
    ):
        leader = RaftNode(1, [2, 3])
        leader.role = Role.LEADER
        leader.current_term = 2
        leader.log.append(LogEntry(term=1, index=1, command="from-old-term"))
        leader.next_index = {2: 2, 3: 2}
        leader.match_index = {1: 1, 2: 0, 3: 0}

        # Both followers ack replicating index 1 (term 1) -- a majority has
        # it, but it's not from the leader's current term, so it must not
        # be committed by this alone.
        from raftkv.raft.rpc import AppendEntriesReply

        reply_from_2 = AppendEntriesReply(term=2, success=True, follower_id=2, match_index=1)
        leader.step(Message(2, 1, reply_from_2))
        self.assertEqual(leader.commit_index, 0)

        reply_from_3 = AppendEntriesReply(term=2, success=True, follower_id=3, match_index=1)
        leader.step(Message(3, 1, reply_from_3))
        self.assertEqual(
            leader.commit_index, 0, "must not commit a prior-term entry via majority alone"
        )


if __name__ == "__main__":
    unittest.main()
