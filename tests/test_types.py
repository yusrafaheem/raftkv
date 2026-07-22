import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from raftkv.raft.types import Role


class TestRoleEnum(unittest.TestCase):
    def test_there_are_exactly_the_three_roles_the_paper_defines(self):
        self.assertEqual({r.value for r in Role}, {"follower", "candidate", "leader"})

    def test_role_is_a_str_subclass_so_it_serializes_and_compares_naturally(self):
        self.assertEqual(Role.LEADER, "leader")
        self.assertIsInstance(Role.LEADER, str)


class TestPackageVersion(unittest.TestCase):
    def test_package_exposes_a_version_string(self):
        import raftkv

        self.assertRegex(raftkv.__version__, r"^\d+\.\d+\.\d+$")


if __name__ == "__main__":
    unittest.main()
