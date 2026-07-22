"""
Scenario-driven fault-injection tests: the specific failure modes a
Raft-backed store is supposed to survive without losing or corrupting
committed data -- a crashed leader mid-write, a network partition, a
node that comes back after missing a stretch of the log. Where
`test_safety.py` checks invariants across many random seeds,
this file checks concrete, named scenarios end to end (propose ->
commit -> verify durability/availability), since those are the stories
that actually matter when explaining what this project demonstrates.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from raftkv.cluster import SimulatedCluster
from raftkv.kv.store import Command, SetCommand
from raftkv.raft.types import Role


def elected_cluster(node_ids, seed):
    c = SimulatedCluster(list(node_ids), seed=seed)
    ok = c.run_until(lambda cl: cl.leader() is not None, max_ticks=300)
    assert ok
    return c


class TestLeaderCrashMidWrite(unittest.TestCase):
    def test_write_proposed_but_not_yet_committed_when_leader_dies_may_be_lost_but_cluster_recovers(
        self,
    ):
        c = elected_cluster([1, 2, 3, 4, 5], seed=1)
        leader = c.leader()
        # Cut the leader off from everyone *before* it can replicate this
        # particular write -- the write only lands in the leader's own
        # log, never reaches a majority, and is correctly abandoned once
        # a new leader (which never saw it) takes over.
        c.partition([{leader}, set(n for n in c.node_ids if n != leader)])
        c.propose(Command("c", 1, SetCommand("x", "lost")), via=leader)
        c.kill(leader)
        c.heal()

        ok = c.run_until(lambda cl: cl.leader() is not None, max_ticks=300)
        self.assertTrue(ok)
        new_leader = c.leader()
        self.assertNotEqual(new_leader, leader)
        # the abandoned write must not have made it into the new leader's log
        self.assertIsNone(c.state_machines[new_leader].get("x"))

    def test_write_committed_before_leader_dies_survives_and_stays_visible(self):
        c = elected_cluster([1, 2, 3, 4, 5], seed=2)
        leader = c.leader()
        result = c.propose(Command("c", 1, SetCommand("x", "durable")), via=leader)
        ok = c.run_until(lambda cl: cl.is_committed_everywhere(result.index), max_ticks=200)
        self.assertTrue(ok)

        c.kill(leader)
        ok = c.run_until(lambda cl: cl.leader() is not None, max_ticks=300)
        self.assertTrue(ok)
        new_leader = c.leader()
        self.assertNotEqual(new_leader, leader)

        result2 = c.propose(Command("c", 2, SetCommand("y", "also-durable")), via=new_leader)
        c.run_until(lambda cl: cl.is_committed_everywhere(result2.index), max_ticks=200)
        for n in c.alive:
            self.assertEqual(c.state_machines[n].get("x"), "durable")
            self.assertEqual(c.state_machines[n].get("y"), "also-durable")

    def test_cluster_survives_repeated_leader_kills_and_keeps_making_progress(self):
        c = elected_cluster([1, 2, 3, 4, 5], seed=3)
        written = []
        for i in range(1, 6):
            leader = c.leader()
            if leader is None:
                c.run_until(lambda cl: cl.leader() is not None, max_ticks=300)
                leader = c.leader()
            result = c.propose(Command("c", i, SetCommand(f"k{i}", f"v{i}")), via=leader)
            ok = c.run_until(
                lambda cl, idx=result.index: cl.is_committed_everywhere(idx), max_ticks=200
            )
            self.assertTrue(ok, f"write {i} failed to commit")
            written.append((f"k{i}", f"v{i}"))
            if len(c.alive) > 3:  # keep a majority alive so progress remains possible
                c.kill(leader)
                c.run_until(lambda cl: cl.leader() is not None, max_ticks=300)

        for key, value in written:
            for n in c.alive:
                self.assertEqual(c.state_machines[n].get(key), value)


class TestNetworkPartition(unittest.TestCase):
    def test_minority_partition_cannot_commit_writes(self):
        c = elected_cluster([1, 2, 3, 4, 5], seed=4)
        leader = c.leader()
        follower = next(n for n in c.node_ids if n != leader)
        minority = {leader, follower}
        majority = set(c.node_ids) - minority
        c.partition([minority, majority])

        result = c.propose(Command("c", 1, SetCommand("x", "1")), via=leader)
        self.assertIsNotNone(result.index)
        c.run(100)
        self.assertFalse(c.nodes[leader].commit_index >= result.index)

    def test_majority_partition_keeps_making_progress_during_a_partition(self):
        c = elected_cluster([1, 2, 3, 4, 5], seed=5)
        leader = c.leader()
        follower = next(n for n in c.node_ids if n != leader)
        minority = {leader, follower}
        majority = set(c.node_ids) - minority
        c.partition([minority, majority])

        ok = c.run_until(
            lambda cl: any(cl.nodes[n].role.value == "leader" for n in majority), max_ticks=400
        )
        self.assertTrue(ok)
        majority_leader = next(n for n in majority if c.nodes[n].role.value == "leader")
        result = c.propose(Command("c", 1, SetCommand("x", "progress")), via=majority_leader)
        ok = c.run_until(
            lambda cl: all(cl.nodes[n].commit_index >= result.index for n in majority),
            max_ticks=300,
        )
        self.assertTrue(ok)

    def test_healed_partition_reconverges_to_a_single_leader_and_a_consistent_log(self):
        c = elected_cluster([1, 2, 3, 4, 5], seed=6)
        leader = c.leader()
        follower = next(n for n in c.node_ids if n != leader)
        minority = {leader, follower}
        majority = set(c.node_ids) - minority
        c.partition([minority, majority])
        c.run_until(
            lambda cl: any(cl.nodes[n].role.value == "leader" for n in majority), max_ticks=400
        )
        majority_leader = next(n for n in majority if c.nodes[n].role.value == "leader")
        result = c.propose(Command("c", 1, SetCommand("x", "winner")), via=majority_leader)
        c.run_until(
            lambda cl: all(cl.nodes[n].commit_index >= result.index for n in majority),
            max_ticks=300,
        )

        c.heal()
        ok = c.run_until(lambda cl: len(cl.leaders()) == 1, max_ticks=400)
        self.assertTrue(ok)
        ok = c.run_until(lambda cl: cl.is_committed_everywhere(result.index), max_ticks=300)
        self.assertTrue(ok)
        for n in c.node_ids:
            self.assertEqual(c.state_machines[n].get("x"), "winner")


class TestNodeRevival(unittest.TestCase):
    def test_a_node_that_missed_a_long_stretch_of_writes_catches_up_after_revival(self):
        c = elected_cluster([1, 2, 3], seed=7)
        laggard = next(n for n in c.node_ids if n != c.leader())
        c.kill(laggard)

        last_result = None
        for i in range(1, 11):
            leader = c.leader()
            last_result = c.propose(Command("c", i, SetCommand("k", str(i))), via=leader)
            c.run_until(
                lambda cl, idx=last_result.index: cl.is_committed_everywhere(idx), max_ticks=200
            )

        self.assertNotIn("k", c.state_machines[laggard].snapshot())  # never applied while dead

        c.revive(laggard)
        ok = c.run_until(
            lambda cl, idx=last_result.index: laggard in cl.alive
            and cl.nodes[laggard].commit_index >= idx,
            max_ticks=300,
        )
        self.assertTrue(ok)
        c.run(20)  # give take_committed_entries a chance to apply the catch-up
        self.assertEqual(c.state_machines[laggard].get("k"), "10")


class TestRepeatedPartitionAndHeal(unittest.TestCase):
    def test_cluster_keeps_making_progress_across_several_partition_heal_cycles(self):
        c = elected_cluster([1, 2, 3, 4, 5], seed=9)
        for i in range(1, 4):
            leader = c.leader()
            follower = next(n for n in c.node_ids if n != leader)
            minority = {leader, follower}
            majority = set(c.node_ids) - minority
            c.partition([minority, majority])
            c.run_until(
                lambda cl: any(cl.nodes[n].role.value == "leader" for n in majority),
                max_ticks=400,
            )
            majority_leader = next(n for n in majority if c.nodes[n].role.value == "leader")
            result = c.propose(Command("c", i, SetCommand(f"k{i}", f"v{i}")), via=majority_leader)
            ok = c.run_until(
                lambda cl, idx=result.index: all(
                    cl.nodes[n].commit_index >= idx for n in majority
                ),
                max_ticks=300,
            )
            self.assertTrue(ok, f"cycle {i} failed to commit during the partition")
            c.heal()
            c.run_until(lambda cl: len(cl.leaders()) == 1, max_ticks=400)


class TestSimultaneousKills(unittest.TestCase):
    def test_cluster_recovers_after_two_non_leader_nodes_die_at_once(self):
        c = elected_cluster([1, 2, 3, 4, 5], seed=8)
        leader = c.leader()
        victims = [n for n in c.node_ids if n != leader][:2]
        for v in victims:
            c.kill(v)

        result = c.propose(Command("c", 1, SetCommand("x", "1")), via=leader)
        ok = c.run_until(lambda cl: cl.is_committed_everywhere(result.index), max_ticks=200)
        self.assertTrue(ok)
        for n in c.alive:
            self.assertEqual(c.state_machines[n].get("x"), "1")


class TestLeaderIsolatedAlone(unittest.TestCase):
    def test_a_leader_partitioned_off_by_itself_cannot_commit_and_a_new_leader_takes_over(self):
        c = elected_cluster([1, 2, 3, 4, 5], seed=6)
        old_leader = c.leader()
        others = [n for n in c.node_ids if n != old_leader]
        c.partition([{old_leader}, set(others)])

        c.propose(Command("c", 1, SetCommand("x", "1")), via=old_leader)
        c.run(50)
        self.assertEqual(c.nodes[old_leader].commit_index, 0)

        def a_new_leader_emerged_among_the_majority(cl):
            return any(cl.nodes[n].role is Role.LEADER for n in others)

        ok = c.run_until(a_new_leader_emerged_among_the_majority, max_ticks=300)
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
