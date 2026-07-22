import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from raftkv.cluster import SimulatedCluster
from raftkv.raft.log import LogEntry
from raftkv.raft.node import RaftNode
from raftkv.raft.rpc import AppendEntriesArgs, Message, RequestVoteArgs, RequestVoteReply
from raftkv.raft.types import Role


class TestSingleNodeCluster(unittest.TestCase):
    def test_single_node_becomes_leader_on_first_election(self):
        node = RaftNode(1, [], rng=random.Random(0))
        messages = node.tick()
        while node.role is not Role.LEADER:
            messages = node.tick()
        self.assertEqual(node.role, Role.LEADER)
        self.assertEqual(node.current_term, 1)
        self.assertEqual(messages, [])  # no peers to send heartbeats to


class TestSingleNodeClusterWithExistingLog(unittest.TestCase):
    def test_single_node_still_becomes_leader_even_with_a_pre_existing_log(self):
        node = RaftNode(1, [], rng=random.Random(0))
        node.log.append(LogEntry(term=0, index=1, command="pre-existing"))
        while node.role is not Role.LEADER:
            node.tick()
        self.assertEqual(node.role, Role.LEADER)
        self.assertEqual(node.log.last_index, 1)  # untouched by becoming leader


class TestThreeNodeElection(unittest.TestCase):
    def test_a_leader_is_elected_within_a_bounded_number_of_ticks(self):
        c = SimulatedCluster([1, 2, 3], seed=1)
        ok = c.run_until(lambda cl: cl.leader() is not None, max_ticks=200)
        self.assertTrue(ok)
        self.assertEqual(len(c.leaders()), 1)

    def test_the_elected_leader_and_all_followers_agree_on_the_term(self):
        c = SimulatedCluster([1, 2, 3], seed=2)
        c.run_until(lambda cl: cl.leader() is not None, max_ticks=200)
        leader_term = c.nodes[c.leader()].current_term
        for node_id in c.node_ids:
            self.assertEqual(c.nodes[node_id].current_term, leader_term)

    def test_election_is_reproducible_given_the_same_seed(self):
        leaders = []
        for _ in range(3):
            c = SimulatedCluster([1, 2, 3], seed=42)
            c.run_until(lambda cl: cl.leader() is not None, max_ticks=200)
            leaders.append((c.leader(), c.nodes[c.leader()].current_term))
        self.assertEqual(len(set(leaders)), 1)  # every run picked the same leader/term

    def test_many_seeds_all_converge_to_exactly_one_leader(self):
        for seed in range(30):
            with self.subTest(seed=seed):
                c = SimulatedCluster([1, 2, 3, 4, 5], seed=seed)
                ok = c.run_until(lambda cl: cl.leader() is not None, max_ticks=300)
                self.assertTrue(ok, f"no leader elected for seed {seed}")
                self.assertEqual(len(c.leaders()), 1)


class TestRequestVoteRpcSemantics(unittest.TestCase):
    """Direct, unit-level tests of RaftNode.step() for RequestVote, without
    going through a whole simulated cluster -- these pin down the exact
    Figure 2 rules rather than just observing emergent behavior."""

    def test_grants_vote_when_log_is_up_to_date_and_has_not_voted_yet(self):
        node = RaftNode(1, [2, 3])
        args = RequestVoteArgs(term=1, candidate_id=2, last_log_index=0, last_log_term=0)
        replies = node.step(Message(2, 1, args))
        self.assertEqual(len(replies), 1)
        self.assertTrue(replies[0].payload.vote_granted)
        self.assertEqual(node.voted_for, 2)

    def test_rejects_vote_for_a_stale_term(self):
        node = RaftNode(1, [2, 3])
        node.current_term = 5
        args = RequestVoteArgs(term=3, candidate_id=2, last_log_index=0, last_log_term=0)
        reply = node.step(Message(2, 1, args))[0]
        self.assertFalse(reply.payload.vote_granted)

    def test_rejects_a_second_vote_request_in_the_same_term(self):
        node = RaftNode(1, [2, 3])
        vote_for_2 = RequestVoteArgs(term=1, candidate_id=2, last_log_index=0, last_log_term=0)
        node.step(Message(2, 1, vote_for_2))
        vote_for_3 = RequestVoteArgs(term=1, candidate_id=3, last_log_index=0, last_log_term=0)
        reply = node.step(Message(3, 1, vote_for_3))[0]
        self.assertFalse(reply.payload.vote_granted)
        self.assertEqual(node.voted_for, 2)

    def test_grants_a_repeat_vote_for_the_same_candidate_in_the_same_term(self):
        # e.g. a duplicated/retried RequestVote RPC.
        node = RaftNode(1, [2, 3])
        args = RequestVoteArgs(term=1, candidate_id=2, last_log_index=0, last_log_term=0)
        node.step(Message(2, 1, args))
        reply = node.step(Message(2, 1, args))[0]
        self.assertTrue(reply.payload.vote_granted)

    def test_grants_vote_when_candidate_log_has_same_last_term_and_is_at_least_as_long(self):
        node = RaftNode(1, [2, 3])
        node.current_term = 3
        node.log.append(LogEntry(term=2, index=1, command="x"))
        args = RequestVoteArgs(term=4, candidate_id=2, last_log_index=1, last_log_term=2)
        reply = node.step(Message(2, 1, args))[0]
        self.assertTrue(reply.payload.vote_granted)

    def test_rejects_vote_when_candidate_log_is_less_up_to_date(self):
        node = RaftNode(1, [2, 3])
        node.current_term = 3
        node.log.append(LogEntry(term=3, index=1, command="x"))
        # candidate's last_log_term (2) is behind ours (3) -> reject
        args = RequestVoteArgs(term=4, candidate_id=2, last_log_index=1, last_log_term=2)
        reply = node.step(Message(2, 1, args))[0]
        self.assertFalse(reply.payload.vote_granted)

    def test_higher_term_in_request_vote_causes_step_down_and_term_adoption(self):
        node = RaftNode(1, [2, 3])
        node.role = Role.LEADER
        node.current_term = 2
        args = RequestVoteArgs(term=5, candidate_id=2, last_log_index=0, last_log_term=0)
        node.step(Message(2, 1, args))
        self.assertEqual(node.role, Role.FOLLOWER)
        self.assertEqual(node.current_term, 5)

    def test_stale_vote_reply_from_an_earlier_term_is_ignored(self):
        node = RaftNode(1, [2, 3], rng=random.Random(0))
        node.role = Role.CANDIDATE
        node.current_term = 5
        node.votes_received = {1}
        stale_reply = RequestVoteReply(term=3, vote_granted=True, voter_id=2)
        out = node.step(Message(2, 1, stale_reply))
        self.assertEqual(out, [])
        self.assertEqual(node.votes_received, {1})

    def test_vote_reply_after_role_changed_away_from_candidate_is_ignored(self):
        node = RaftNode(1, [2, 3], rng=random.Random(0))
        node.role = Role.CANDIDATE
        node.current_term = 1
        node.votes_received = {1}
        node.role = Role.FOLLOWER  # e.g. we just saw a legitimate leader's heartbeat
        reply = RequestVoteReply(term=1, vote_granted=True, voter_id=2)
        out = node.step(Message(2, 1, reply))
        self.assertEqual(out, [])


class TestVoteReplyIgnoredOnceAlreadyLeader(unittest.TestCase):
    def test_a_late_vote_reply_after_already_winning_the_election_is_a_harmless_no_op(self):
        node = RaftNode(1, [2, 3], rng=random.Random(0))
        node.role = Role.CANDIDATE
        node.current_term = 1
        node.votes_received = {1, 2}  # already has a majority, e.g. from node 2's reply
        node._become_leader()
        late_reply = RequestVoteReply(term=1, vote_granted=True, voter_id=3)
        out = node.step(Message(3, 1, late_reply))
        self.assertEqual(out, [])
        self.assertEqual(node.role, Role.LEADER)


class TestAppendEntriesCausesElectionStepDown(unittest.TestCase):
    def test_candidate_steps_down_on_append_entries_from_legitimate_leader(self):
        node = RaftNode(1, [2, 3], rng=random.Random(0))
        node.role = Role.CANDIDATE
        node.current_term = 3
        args = AppendEntriesArgs(
            term=3, leader_id=2, prev_log_index=0, prev_log_term=0, entries=(), leader_commit=0
        )
        node.step(Message(2, 1, args))
        self.assertEqual(node.role, Role.FOLLOWER)
        self.assertEqual(node.leader_id, 2)


if __name__ == "__main__":
    unittest.main()
