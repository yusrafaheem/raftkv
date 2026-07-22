"""Direct unit tests of the RPC dataclasses themselves (`raftkv.raft.rpc`),
as distinct from `test_codec.py`'s wire round-trip tests -- these just
check the shapes and equality semantics the rest of the codebase leans
on without ever going through JSON."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from raftkv.raft.log import LogEntry
from raftkv.raft.rpc import AppendEntriesArgs, Message, RequestVoteArgs


class TestMessageEnvelope(unittest.TestCase):
    def test_two_messages_with_identical_fields_are_equal(self):
        args = RequestVoteArgs(term=1, candidate_id=2, last_log_index=0, last_log_term=0)
        a = Message(1, 2, args)
        b = Message(1, 2, args)
        self.assertEqual(a, b)

    def test_messages_with_different_destinations_are_not_equal(self):
        args = RequestVoteArgs(term=1, candidate_id=2, last_log_index=0, last_log_term=0)
        self.assertNotEqual(Message(1, 2, args), Message(1, 3, args))

    def test_message_is_frozen(self):
        args = RequestVoteArgs(term=1, candidate_id=2, last_log_index=0, last_log_term=0)
        msg = Message(1, 2, args)
        with self.assertRaises(AttributeError):
            msg.dst = 99  # type: ignore[misc]


class TestAppendEntriesArgsShape(unittest.TestCase):
    def test_entries_tuple_is_immutable_and_order_preserving(self):
        e1 = LogEntry(term=1, index=1, command="a")
        e2 = LogEntry(term=1, index=2, command="b")
        args = AppendEntriesArgs(
            term=1,
            leader_id=1,
            prev_log_index=0,
            prev_log_term=0,
            entries=(e1, e2),
            leader_commit=0,
        )
        self.assertEqual(args.entries, (e1, e2))
        with self.assertRaises(AttributeError):
            args.entries = ()  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
