"""
raftkv
========

A from-scratch implementation of the Raft consensus algorithm and a
replicated key-value store built on top of it.

The design deliberately follows the same "consensus is a library, not a
service" architecture used by production implementations (etcd/raft,
TiKV's raft-rs, Hashicorp's raft): `raftkv.raft.node.RaftNode` is a pure
state machine with no I/O, no threads, and no clock of its own. It
exposes three entry points -- `tick()`, `step(message)`, and
`propose(command)` -- each of which returns the outbound messages the
caller should deliver. Everything about *when* time advances and *how*
messages get from one node to another is somebody else's problem: a
deterministic in-process simulator for tests (`raftkv.transport.simulated`),
or a real asyncio TCP transport for the runnable demo cluster
(`raftkv.transport.tcp`).

That separation is what makes the consensus logic itself exhaustively
and deterministically testable -- see `tests/test_safety.py` and
`tests/test_fault_tolerance.py`, which run thousands of simulated ticks
with injected leader kills and network partitions, all with zero
`time.sleep()` and zero flakiness.
"""

__version__ = "0.1.0"
