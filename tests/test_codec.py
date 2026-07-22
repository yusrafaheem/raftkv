import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from raftkv.kv.client import ClientRequest, ClientResponse
from raftkv.kv.store import Command, CompareAndSwapCommand, DeleteCommand, GetCommand, SetCommand
from raftkv.raft.log import LogEntry
from raftkv.raft.rpc import (
    AppendEntriesArgs,
    AppendEntriesReply,
    Message,
    RequestVoteArgs,
    RequestVoteReply,
)
from raftkv.transport.codec import (
    decode_client_request,
    decode_client_response,
    decode_message,
    encode_client_request,
    encode_client_response,
    encode_message,
)


class TestOpRoundTrip(unittest.TestCase):
    def _round_trip_command(self, command: Command) -> Command:
        entry = LogEntry(term=1, index=1, command=command)
        from raftkv.transport.codec import decode_entry, encode_entry

        return decode_entry(encode_entry(entry)).command

    def test_get_command_round_trips(self):
        cmd = Command("c", 1, GetCommand("x"))
        self.assertEqual(self._round_trip_command(cmd), cmd)

    def test_set_command_round_trips(self):
        cmd = Command("c", 1, SetCommand("x", "hello"))
        self.assertEqual(self._round_trip_command(cmd), cmd)

    def test_delete_command_round_trips(self):
        cmd = Command("c", 1, DeleteCommand("x"))
        self.assertEqual(self._round_trip_command(cmd), cmd)

    def test_cas_command_round_trips_including_none_expected(self):
        cmd = Command("c", 1, CompareAndSwapCommand("x", None, "v"))
        self.assertEqual(self._round_trip_command(cmd), cmd)
        cmd2 = Command("c", 2, CompareAndSwapCommand("x", "old", "new"))
        self.assertEqual(self._round_trip_command(cmd2), cmd2)


class TestLogEntryRoundTrip(unittest.TestCase):
    def test_a_bare_log_entry_round_trips_through_encode_decode_entry(self):
        from raftkv.transport.codec import decode_entry, encode_entry

        entry = LogEntry(term=3, index=7, command=Command("c", 1, SetCommand("x", "1")))
        self.assertEqual(decode_entry(encode_entry(entry)), entry)


class TestMessageRoundTrip(unittest.TestCase):
    def test_request_vote_args_round_trips(self):
        args = RequestVoteArgs(term=3, candidate_id=1, last_log_index=5, last_log_term=2)
        msg = Message(1, 2, args)
        self.assertEqual(decode_message(encode_message(msg)), msg)

    def test_request_vote_reply_round_trips(self):
        msg = Message(2, 1, RequestVoteReply(term=3, vote_granted=True, voter_id=2))
        self.assertEqual(decode_message(encode_message(msg)), msg)

    def test_append_entries_args_round_trips_with_entries(self):
        entries = (
            LogEntry(term=1, index=1, command=Command("c", 1, SetCommand("a", "1"))),
            LogEntry(term=2, index=2, command=Command("c", 2, DeleteCommand("a"))),
        )
        args = AppendEntriesArgs(
            term=2, leader_id=1, prev_log_index=0, prev_log_term=0, entries=entries, leader_commit=1
        )
        msg = Message(1, 2, args)
        round_tripped = decode_message(encode_message(msg))
        self.assertEqual(round_tripped, msg)

    def test_append_entries_args_round_trips_with_no_entries(self):
        args = AppendEntriesArgs(
            term=1, leader_id=1, prev_log_index=3, prev_log_term=1, entries=(), leader_commit=3
        )
        msg = Message(1, 2, args)
        self.assertEqual(decode_message(encode_message(msg)), msg)

    def test_append_entries_reply_round_trips(self):
        reply = AppendEntriesReply(term=1, success=True, follower_id=2, match_index=5)
        msg = Message(2, 1, reply)
        self.assertEqual(decode_message(encode_message(msg)), msg)

    def test_encoded_message_is_json_serializable(self):
        import json

        args = RequestVoteArgs(term=1, candidate_id=1, last_log_index=0, last_log_term=0)
        msg = Message(1, 2, args)
        encoded = encode_message(msg)
        round_tripped_through_json = json.loads(json.dumps(encoded))
        self.assertEqual(decode_message(round_tripped_through_json), msg)


class TestClientProtocolRoundTrip(unittest.TestCase):
    def test_client_request_round_trips(self):
        req = ClientRequest("client-a", 7, SetCommand("x", "1"))
        self.assertEqual(decode_client_request(encode_client_request(req)), req)

    def test_client_response_round_trips_ok(self):
        resp = ClientResponse(ok=True, result="1")
        self.assertEqual(decode_client_response(encode_client_response(resp)), resp)

    def test_client_response_round_trips_error_with_leader_hint(self):
        resp = ClientResponse(ok=False, leader_hint=3, error="not leader")
        self.assertEqual(decode_client_response(encode_client_response(resp)), resp)

    def test_client_response_round_trips_a_none_result(self):
        # SetCommand and DeleteCommand both return None on success -- make
        # sure that's distinguishable on the wire from "no field sent".
        resp = ClientResponse(ok=True, result=None)
        self.assertEqual(decode_client_response(encode_client_response(resp)), resp)


if __name__ == "__main__":
    unittest.main()
