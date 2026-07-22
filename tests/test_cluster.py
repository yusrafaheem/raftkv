"""Direct unit tests of `SimulatedCluster` itself, as distinct from the
election/replication/safety tests that just use it as a harness."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from raftkv.cluster import SimulatedCluster
from raftkv.kv.store import Command, SetCommand


class TestProposeWithoutALeader(unittest.TestCase):
    def test_propose_with_no_leader_and_no_via_returns_none(self):
        c = SimulatedCluster([1, 2, 3], seed=0)
        # Fresh cluster, tick zero -- no election has happened yet.
        self.assertIsNone(c.leader())
        self.assertIsNone(c.propose(Command("c", 1, SetCommand("x", "1"))))

    def test_propose_via_a_dead_node_returns_none(self):
        c = SimulatedCluster([1, 2, 3], seed=1)
        c.run_until(lambda cl: cl.leader() is not None, max_ticks=200)
        leader = c.leader()
        c.kill(leader)
        self.assertIsNone(c.propose(Command("c", 1, SetCommand("x", "1")), via=leader))


class TestKillAndRevive(unittest.TestCase):
    def test_a_killed_node_is_no_longer_in_alive(self):
        c = SimulatedCluster([1, 2, 3], seed=2)
        c.kill(2)
        self.assertNotIn(2, c.alive)
        self.assertIn(1, c.alive)

    def test_reviving_a_node_that_was_never_killed_is_a_harmless_no_op(self):
        c = SimulatedCluster([1, 2, 3], seed=3)
        c.revive(1)  # never killed
        self.assertEqual(c.alive, {1, 2, 3})

    def test_a_revived_nodes_raft_state_survived_being_dead(self):
        c = SimulatedCluster([1, 2, 3], seed=4)
        c.run_until(lambda cl: cl.leader() is not None, max_ticks=200)
        leader = c.leader()
        c.propose(Command("c", 1, SetCommand("x", "1")), via=leader)
        c.run(20)
        log_length_before = c.nodes[leader].log.last_index
        c.kill(leader)
        c.run(50)  # cluster keeps going without it, elects a new leader
        c.revive(leader)
        # "killing" only stops delivering ticks/messages -- the dead
        # node's own in-memory log must be untouched, not reset.
        self.assertEqual(c.nodes[leader].log.last_index, log_length_before)


class TestLeadersBeforeElection(unittest.TestCase):
    def test_leaders_is_empty_on_a_freshly_constructed_cluster(self):
        c = SimulatedCluster([1, 2, 3], seed=5)
        self.assertEqual(c.leaders(), [])


if __name__ == "__main__":
    unittest.main()
