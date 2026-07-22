import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from raftkv.kv.store import (
    Command,
    CompareAndSwapCommand,
    DeleteCommand,
    GetCommand,
    KVStateMachine,
    SetCommand,
)


class TestBasicOps(unittest.TestCase):
    def setUp(self):
        self.sm = KVStateMachine()

    def test_get_missing_key_is_none(self):
        result = self.sm.apply(Command("c", 1, GetCommand("x")))
        self.assertIsNone(result)

    def test_set_then_get(self):
        self.sm.apply(Command("c", 1, SetCommand("x", "1")))
        self.assertEqual(self.sm.apply(Command("c", 2, GetCommand("x"))), "1")

    def test_set_overwrites(self):
        self.sm.apply(Command("c", 1, SetCommand("x", "1")))
        self.sm.apply(Command("c", 2, SetCommand("x", "2")))
        self.assertEqual(self.sm.get("x"), "2")

    def test_delete_removes_key(self):
        self.sm.apply(Command("c", 1, SetCommand("x", "1")))
        self.sm.apply(Command("c", 2, DeleteCommand("x")))
        self.assertIsNone(self.sm.get("x"))

    def test_delete_missing_key_is_a_harmless_no_op(self):
        self.sm.apply(Command("c", 1, DeleteCommand("nope")))
        self.assertIsNone(self.sm.get("nope"))

    def test_set_after_delete_recreates_the_key(self):
        self.sm.apply(Command("c", 1, SetCommand("x", "1")))
        self.sm.apply(Command("c", 2, DeleteCommand("x")))
        self.sm.apply(Command("c", 3, SetCommand("x", "2")))
        self.assertEqual(self.sm.get("x"), "2")

    def test_applied_count_increases_once_per_new_command(self):
        self.sm.apply(Command("c", 1, SetCommand("x", "1")))
        self.sm.apply(Command("c", 2, SetCommand("y", "2")))
        self.assertEqual(self.sm.applied_count, 2)


class TestCompareAndSwap(unittest.TestCase):
    def setUp(self):
        self.sm = KVStateMachine()

    def test_cas_on_missing_key_with_expected_none_succeeds(self):
        result = self.sm.apply(Command("c", 1, CompareAndSwapCommand("x", None, "1")))
        self.assertTrue(result)
        self.assertEqual(self.sm.get("x"), "1")

    def test_cas_on_missing_key_with_non_none_expected_fails(self):
        result = self.sm.apply(Command("c", 1, CompareAndSwapCommand("x", "anything", "1")))
        self.assertFalse(result)
        self.assertIsNone(self.sm.get("x"))

    def test_cas_with_matching_expected_succeeds_and_updates(self):
        self.sm.apply(Command("c", 1, SetCommand("x", "1")))
        result = self.sm.apply(Command("c", 2, CompareAndSwapCommand("x", "1", "2")))
        self.assertTrue(result)
        self.assertEqual(self.sm.get("x"), "2")

    def test_cas_with_stale_expected_fails_and_leaves_value_untouched(self):
        self.sm.apply(Command("c", 1, SetCommand("x", "1")))
        result = self.sm.apply(Command("c", 2, CompareAndSwapCommand("x", "stale", "2")))
        self.assertFalse(result)
        self.assertEqual(self.sm.get("x"), "1")

    def test_cas_where_new_value_equals_the_current_value_still_counts_as_a_success(self):
        self.sm.apply(Command("c", 1, SetCommand("x", "same")))
        result = self.sm.apply(Command("c", 2, CompareAndSwapCommand("x", "same", "same")))
        self.assertTrue(result)
        self.assertEqual(self.sm.get("x"), "same")

    def test_only_one_of_two_racing_cas_calls_can_win(self):
        self.sm.apply(Command("c", 1, SetCommand("x", "start")))
        first = self.sm.apply(Command("a", 1, CompareAndSwapCommand("x", "start", "from-a")))
        second = self.sm.apply(Command("b", 1, CompareAndSwapCommand("x", "start", "from-b")))
        self.assertTrue(first)
        self.assertFalse(second)
        self.assertEqual(self.sm.get("x"), "from-a")

    def test_cas_expecting_none_fails_once_the_key_has_been_set(self):
        self.sm.apply(Command("c", 1, SetCommand("x", "1")))
        result = self.sm.apply(Command("c", 2, CompareAndSwapCommand("x", None, "2")))
        self.assertFalse(result)
        self.assertEqual(self.sm.get("x"), "1")


class TestClientRequestDeduplication(unittest.TestCase):
    """Section 8 of the Raft paper: applying the same (client_id,
    request_id) command twice must only take effect once, so a client
    that retries after an ambiguous failure can't double-apply a
    non-idempotent command."""

    def setUp(self):
        self.sm = KVStateMachine()

    def test_retried_command_with_same_request_id_is_applied_only_once(self):
        cmd = Command("client-a", 1, CompareAndSwapCommand("x", None, "first"))
        first_result = self.sm.apply(cmd)
        second_result = self.sm.apply(cmd)  # simulates the client retrying
        self.assertTrue(first_result)
        self.assertTrue(second_result)  # cached result, not a fresh (failing) CAS
        self.assertEqual(self.sm.applied_count, 1)
        self.assertEqual(self.sm.get("x"), "first")

    def test_retry_of_a_cas_that_originally_failed_still_returns_the_original_false(self):
        self.sm.apply(Command("c", 1, SetCommand("x", "locked")))
        cmd = Command("client-a", 2, CompareAndSwapCommand("x", "wrong", "new"))
        first = self.sm.apply(cmd)
        second = self.sm.apply(cmd)
        self.assertFalse(first)
        self.assertFalse(second)

    def test_a_lower_or_equal_request_id_from_the_same_client_is_ignored(self):
        self.sm.apply(Command("client-a", 5, SetCommand("x", "v5")))
        self.sm.apply(Command("client-a", 3, SetCommand("x", "v3-should-be-ignored")))
        self.sm.apply(Command("client-a", 5, SetCommand("x", "v5-again-should-be-ignored")))
        self.assertEqual(self.sm.get("x"), "v5")

    def test_a_higher_request_id_from_the_same_client_is_applied_normally(self):
        self.sm.apply(Command("client-a", 1, SetCommand("x", "v1")))
        self.sm.apply(Command("client-a", 2, SetCommand("x", "v2")))
        self.assertEqual(self.sm.get("x"), "v2")
        self.assertEqual(self.sm.applied_count, 2)

    def test_different_clients_have_independent_request_id_sequences(self):
        self.sm.apply(Command("client-a", 1, SetCommand("x", "from-a")))
        self.sm.apply(Command("client-b", 1, SetCommand("y", "from-b")))
        self.assertEqual(self.sm.get("x"), "from-a")
        self.assertEqual(self.sm.get("y"), "from-b")
        self.assertEqual(self.sm.applied_count, 2)


class TestSnapshot(unittest.TestCase):
    def test_snapshot_is_a_copy_not_a_live_view(self):
        sm = KVStateMachine()
        sm.apply(Command("c", 1, SetCommand("x", "1")))
        snap = sm.snapshot()
        sm.apply(Command("c", 2, SetCommand("x", "2")))
        self.assertEqual(snap, {"x": "1"})
        self.assertEqual(sm.get("x"), "2")

    def test_snapshot_of_a_fresh_state_machine_is_empty(self):
        self.assertEqual(KVStateMachine().snapshot(), {})


if __name__ == "__main__":
    unittest.main()
