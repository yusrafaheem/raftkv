"""
raftkv.raft.types
====================

Small shared vocabulary for the consensus module: the three roles a node
can be in (Raft section 5.1, Figure 4), and type aliases for the plain
integers that carry a lot of protocol meaning (`NodeId`, `Term`,
`LogIndex`). These are `int` at runtime -- the aliases exist purely so
signatures read as "this is a term" or "this is a log index" instead of
just "this is an int", which matters a lot in an algorithm where mixing
up a term and an index is an easy, silent way to introduce a
correctness bug.
"""

from __future__ import annotations

from enum import Enum


class Role(str, Enum):
    """A Raft node is always in exactly one of these three roles.

    Valid transitions (Figure 4 of the Raft paper):
      FOLLOWER  -> CANDIDATE   (election timeout elapses)
      CANDIDATE -> CANDIDATE   (election timeout elapses again, new term)
      CANDIDATE -> LEADER      (wins majority of votes)
      CANDIDATE -> FOLLOWER    (discovers current leader or higher term)
      LEADER    -> FOLLOWER    (discovers a higher term)
    There is no direct LEADER -> CANDIDATE transition.
    """

    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


NodeId = int
Term = int
LogIndex = int
