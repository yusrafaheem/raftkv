"""Direct unit tests of `SimulatedNetwork`, isolated from any `RaftNode`
or `SimulatedCluster` -- these just check the delivery/fault-injection
mechanics themselves against plain `Message` objects."""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from raftkv.raft.rpc import Message, RequestVoteReply
from raftkv.transport.simulated import SimulatedNetwork


def msg(src, dst):
    return Message(src, dst, RequestVoteReply(term=1, vote_granted=True, voter_id=src))


class TestBasicDelivery(unittest.TestCase):
    def test_a_message_is_delivered_within_its_delay_range(self):
        net = SimulatedNetwork(random.Random(0), delay_range=(2, 2))
        net.send(msg(1, 2))
        self.assertEqual(net.advance(), [])  # tick 1: not due yet
        self.assertEqual(net.advance(), [msg(1, 2)])  # tick 2: due now

    def test_messages_sent_counter_increments_even_for_dropped_messages(self):
        net = SimulatedNetwork(random.Random(0))
        net.drop_between(1, 2)
        net.send(msg(1, 2))
        self.assertEqual(net.messages_sent, 1)
        self.assertEqual(net.messages_dropped, 1)


class TestDropBetween(unittest.TestCase):
    def test_dropped_pair_blocks_messages_in_either_direction(self):
        net = SimulatedNetwork(random.Random(0))
        net.drop_between(1, 2)
        net.send(msg(1, 2))
        net.send(msg(2, 1))
        self.assertEqual(net.pending_count(), 0)

    def test_restore_between_undoes_a_drop(self):
        net = SimulatedNetwork(random.Random(0))
        net.drop_between(1, 2)
        net.restore_between(1, 2)
        net.send(msg(1, 2))
        self.assertEqual(net.pending_count(), 1)


class TestPartitionWithAnUnlistedNode(unittest.TestCase):
    def test_a_node_missing_from_every_partition_group_is_treated_as_isolated(self):
        net = SimulatedNetwork(random.Random(0))
        net.partition([{1, 2}])  # node 3 is in no group at all
        net.send(msg(1, 3))
        self.assertEqual(net.pending_count(), 0)


class TestNowAdvancesMonotonically(unittest.TestCase):
    def test_now_starts_at_zero_and_increments_once_per_advance(self):
        net = SimulatedNetwork(random.Random(0))
        self.assertEqual(net.now, 0)
        net.advance()
        net.advance()
        net.advance()
        self.assertEqual(net.now, 3)


class TestDropAllPending(unittest.TestCase):
    def test_drop_all_pending_clears_the_queue_and_counts_as_dropped(self):
        net = SimulatedNetwork(random.Random(0))
        net.send(msg(1, 2))
        net.send(msg(1, 3))
        net.drop_all_pending()
        self.assertEqual(net.pending_count(), 0)
        self.assertEqual(net.messages_dropped, 2)


if __name__ == "__main__":
    unittest.main()
