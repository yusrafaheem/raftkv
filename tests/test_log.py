import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from raftkv.raft.log import LogEntry, RaftLog


def entry(index, term, command="cmd"):
    return LogEntry(term=term, index=index, command=command)


class TestEmptyLog(unittest.TestCase):
    def test_last_index_and_term_are_zero(self):
        log = RaftLog()
        self.assertEqual(log.last_index, 0)
        self.assertEqual(log.last_term(), 0)

    def test_term_at_zero_or_missing_index_is_zero(self):
        log = RaftLog()
        self.assertEqual(log.term_at(0), 0)
        self.assertEqual(log.term_at(1), 0)
        self.assertEqual(log.term_at(-1), 0)

    def test_get_missing_index_returns_none(self):
        log = RaftLog()
        self.assertIsNone(log.get(1))

    def test_entries_from_empty_log_is_empty(self):
        log = RaftLog()
        self.assertEqual(log.entries_from(1), [])


class TestAppend(unittest.TestCase):
    def test_append_grows_last_index_and_term(self):
        log = RaftLog()
        log.append(entry(1, term=1))
        self.assertEqual(log.last_index, 1)
        self.assertEqual(log.last_term(), 1)
        log.append(entry(2, term=1))
        log.append(entry(3, term=2))
        self.assertEqual(log.last_index, 3)
        self.assertEqual(log.last_term(), 2)

    def test_append_out_of_order_index_raises(self):
        log = RaftLog()
        log.append(entry(1, term=1))
        with self.assertRaises(ValueError):
            log.append(entry(3, term=1))  # skips index 2

    def test_append_duplicate_index_raises(self):
        log = RaftLog()
        log.append(entry(1, term=1))
        with self.assertRaises(ValueError):
            log.append(entry(1, term=1))

    def test_len_matches_last_index(self):
        log = RaftLog()
        for i in range(1, 6):
            log.append(entry(i, term=1))
        self.assertEqual(len(log), 5)
        self.assertEqual(log.last_index, 5)


class TestTermAtAndGet(unittest.TestCase):
    def setUp(self):
        self.log = RaftLog()
        self.log.append(entry(1, term=1))
        self.log.append(entry(2, term=1))
        self.log.append(entry(3, term=2))

    def test_term_at_returns_the_right_term_per_index(self):
        self.assertEqual(self.log.term_at(1), 1)
        self.assertEqual(self.log.term_at(2), 1)
        self.assertEqual(self.log.term_at(3), 2)

    def test_term_at_past_end_is_zero(self):
        self.assertEqual(self.log.term_at(4), 0)

    def test_get_returns_the_entry(self):
        e = self.log.get(3)
        self.assertEqual((e.term, e.index), (2, 3))

    def test_get_past_end_returns_none(self):
        self.assertIsNone(self.log.get(99))


class TestSlicing(unittest.TestCase):
    def setUp(self):
        self.log = RaftLog()
        for i in range(1, 6):
            self.log.append(entry(i, term=1, command=f"cmd{i}"))

    def test_entries_from_middle(self):
        got = [e.command for e in self.log.entries_from(3)]
        self.assertEqual(got, ["cmd3", "cmd4", "cmd5"])

    def test_entries_from_zero_or_negative_returns_everything(self):
        got = [e.command for e in self.log.entries_from(0)]
        self.assertEqual(got, ["cmd1", "cmd2", "cmd3", "cmd4", "cmd5"])

    def test_entries_from_past_end_is_empty(self):
        self.assertEqual(self.log.entries_from(10), [])

    def test_entries_between_inclusive_range(self):
        got = [e.command for e in self.log.entries_between(2, 4)]
        self.assertEqual(got, ["cmd2", "cmd3", "cmd4"])

    def test_entries_between_clamps_start_below_one(self):
        got = [e.command for e in self.log.entries_between(-5, 2)]
        self.assertEqual(got, ["cmd1", "cmd2"])

    def test_entries_between_end_past_log_length_clamps_to_the_end(self):
        got = [e.command for e in self.log.entries_between(4, 99)]
        self.assertEqual(got, ["cmd4", "cmd5"])

    def test_entries_between_start_greater_than_end_is_empty(self):
        self.assertEqual(self.log.entries_between(4, 2), [])


class TestTruncate(unittest.TestCase):
    def setUp(self):
        self.log = RaftLog()
        for i in range(1, 6):
            self.log.append(entry(i, term=1))

    def test_truncate_from_middle_drops_that_index_and_everything_after(self):
        self.log.truncate_from(3)
        self.assertEqual(self.log.last_index, 2)
        self.assertIsNone(self.log.get(3))

    def test_truncate_from_one_clears_the_whole_log(self):
        self.log.truncate_from(1)
        self.assertEqual(self.log.last_index, 0)

    def test_truncate_from_zero_or_below_clears_the_whole_log(self):
        self.log.truncate_from(0)
        self.assertEqual(self.log.last_index, 0)

    def test_truncate_from_past_end_is_a_no_op(self):
        self.log.truncate_from(10)
        self.assertEqual(self.log.last_index, 5)

    def test_can_append_again_after_truncating(self):
        self.log.truncate_from(3)
        self.log.append(entry(3, term=2, command="replacement"))
        self.assertEqual(self.log.last_index, 3)
        self.assertEqual(self.log.get(3).command, "replacement")


if __name__ == "__main__":
    unittest.main()
