"""
raftkv.transport.simulated
=============================

A deterministic, in-process network for testing `RaftNode` clusters
without any real threading, sockets, or `time.sleep()`.

Time in this module is purely logical: a "tick" is one call to
`SimulatedNetwork.advance()`, and every queued message has an integer
delivery tick computed when it was sent. Two runs seeded with the same
`random.Random` therefore produce byte-for-byte identical message
orderings and node states -- which is what makes it possible to write
fault-injection tests (kill a leader mid-write, partition the network,
drop messages) that are reproducible and never flaky under CI's shared,
variable-speed runners. This mirrors the same philosophy as minirel's
`test_wal_recovery.py` ("simulate a crash" without an actual process
crash) and `test_btree.py` (randomized stress tests against a reference
model) -- prefer a deterministic simulation of the hard-to-trigger case
over trying to actually trigger it.

Faults supported:
  - variable message delay (`delay_range`), so messages can arrive
    out of order;
  - network partitions (`partition()` / `heal()`), which silently drop
    any message crossing a partition boundary -- this is what Raft's
    safety properties are specifically designed to survive;
  - unconditional message drops between two specific nodes
    (`drop_between()` / `restore_between()`), for testing packet loss
    independent of a full partition.
"""

from __future__ import annotations

import heapq
import itertools
import random

from ..raft.rpc import Message
from ..raft.types import NodeId


class SimulatedNetwork:
    def __init__(self, rng: random.Random, delay_range: tuple[int, int] = (1, 3)) -> None:
        self._rng = rng
        self._delay_range = delay_range
        self._now = 0
        self._seq = itertools.count()
        self._queue: list[tuple[int, int, Message]] = []  # (deliver_at, seq, message)
        self._partition_groups: list[frozenset[NodeId]] | None = None
        self._dropped_pairs: set[frozenset[NodeId]] = set()
        self.messages_sent = 0
        self.messages_dropped = 0

    @property
    def now(self) -> int:
        return self._now

    # -- fault injection ---------------------------------------------------

    def partition(self, groups: list[set[NodeId] | frozenset[NodeId]]) -> None:
        """Split the network into isolated groups: messages between two
        nodes in *different* groups are silently dropped; messages within
        the same group are delivered normally. Every node must appear in
        exactly one group."""
        self._partition_groups = [frozenset(g) for g in groups]

    def heal(self) -> None:
        """Remove all partitions -- the network is fully connected again."""
        self._partition_groups = None

    def drop_between(self, a: NodeId, b: NodeId) -> None:
        self._dropped_pairs.add(frozenset((a, b)))

    def restore_between(self, a: NodeId, b: NodeId) -> None:
        self._dropped_pairs.discard(frozenset((a, b)))

    def _is_blocked(self, src: NodeId, dst: NodeId) -> bool:
        if frozenset((src, dst)) in self._dropped_pairs:
            return True
        if self._partition_groups is None:
            return False
        src_group = next((g for g in self._partition_groups if src in g), None)
        dst_group = next((g for g in self._partition_groups if dst in g), None)
        return src_group is not dst_group

    # -- sending / delivery --------------------------------------------------

    def send(self, message: Message) -> None:
        self.messages_sent += 1
        if self._is_blocked(message.src, message.dst):
            self.messages_dropped += 1
            return
        delay = self._rng.randint(*self._delay_range)
        heapq.heappush(self._queue, (self._now + delay, next(self._seq), message))

    def send_all(self, messages: list[Message]) -> None:
        for m in messages:
            self.send(m)

    def advance(self) -> list[Message]:
        """Move logical time forward by one tick and return every message
        now due for delivery (in the deterministic order they were
        scheduled to arrive)."""
        self._now += 1
        delivered = []
        while self._queue and self._queue[0][0] <= self._now:
            _, _, message = heapq.heappop(self._queue)
            delivered.append(message)
        return delivered

    def pending_count(self) -> int:
        return len(self._queue)

    def drop_all_pending(self) -> None:
        """Discard every message currently in flight -- useful for
        simulating a node crashing with unsent/unacked messages still
        queued against it."""
        self.messages_dropped += len(self._queue)
        self._queue.clear()
