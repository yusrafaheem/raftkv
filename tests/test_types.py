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

    def test_role_members_are_distinct_from_one_another(self):
        self.assertNotEqual(Role.FOLLOWER, Role.CANDIDATE)
        self.assertNotEqual(Role.CANDIDATE, Role.LEADER)
        self.assertNotEqual(Role.FOLLOWER, Role.LEADER)


class TestPackageVersion(unittest.TestCase):
    def test_package_exposes_a_version_string(self):
        import raftkv

        self.assertRegex(raftkv.__version__, r"^\d+\.\d+\.\d+$")

    def test_package_version_is_importable_directly_from_the_top_level_package(self):
        from raftkv import __version__

        self.assertIsInstance(__version__, str)


if __name__ == "__main__":
    unittest.main()
