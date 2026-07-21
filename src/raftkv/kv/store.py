"""
raftkv.kv.store
==================

The state machine that sits on top of Raft: a plain in-memory key-value
map, mutated only by applying committed log entries in order.

Section 8 of the Raft paper ("Client interaction") points out a subtlety
that's easy to miss: a client that doesn't hear back from a leader
(because it crashed, or the connection dropped, or it stepped down
mid-request) can't tell whether its command was actually committed or
not, so it must retry -- but if the original command *did* commit, a
naive retry would apply it a second time. `SET x 5` is idempotent and
survives that fine; `INCREMENT x` or `APPEND x ","` would double-apply
and silently corrupt the store.

`KVStateMachine` closes that gap the way the paper suggests: every
command carries a `(client_id, request_id)` pair, and the state machine
remembers the highest `request_id` it has applied per client along with
that request's result. Applying a command whose `request_id` is `<=`
the last one seen for that client is a no-op that just replays the
cached result -- so a client can safely retry an in-flight command as
many times as it wants and it will take effect at most once.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GetCommand:
    """Reads go through the replicated log too, exactly like writes.

    This is the simplest possible way to get linearizable reads out of a
    Raft-backed store: since every command (read or write) is linearized
    by its position in the committed log, a `GetCommand` sees the effect
    of every write ordered before it and none ordered after -- no extra
    protocol needed. The well-known cost is throughput: a read pays a
    full replication round-trip instead of being served locally off the
    leader's state. Production systems avoid that with a ReadIndex or
    lease-based read-only optimization (Raft paper section 8); raftkv
    deliberately doesn't implement either -- see the README's scope
    notes for why that trade-off was made here.
    """

    key: str


@dataclass(frozen=True, slots=True)
class SetCommand:
    key: str
    value: str


@dataclass(frozen=True, slots=True)
class DeleteCommand:
    key: str


@dataclass(frozen=True, slots=True)
class CompareAndSwapCommand:
    """Set `key` to `new_value` only if its current value equals
    `expected`. `expected=None` means "only if the key does not
    currently exist" (a compare-and-*insert*)."""

    key: str
    expected: str | None
    new_value: str


Op = GetCommand | SetCommand | DeleteCommand | CompareAndSwapCommand


@dataclass(frozen=True, slots=True)
class Command:
    """A state-machine command wrapped with the client dedup metadata
    described in this module's docstring. `client_id` should be stable
    for the lifetime of a client session; `request_id` must be strictly
    increasing per client (a simple counter is sufficient)."""

    client_id: str
    request_id: int
    op: Op


class KVStateMachine:
    def __init__(self) -> None:
        self._data: dict[str, str] = {}
        self._last_request: dict[str, tuple[int, object]] = {}  # client_id -> (request_id, result)
        self.applied_count = 0

    def apply(self, command: Command) -> object:
        """Apply a committed `Command`, returning its result. Safe to call
        with a command whose `request_id` has already been applied for
        that `client_id` -- the cached result is returned and the store
        is left untouched (see module docstring)."""
        cached = self._last_request.get(command.client_id)
        if cached is not None and command.request_id <= cached[0]:
            return cached[1]

        result = self._apply_op(command.op)
        self._last_request[command.client_id] = (command.request_id, result)
        self.applied_count += 1
        return result

    def _apply_op(self, op: Op) -> object:
        if isinstance(op, GetCommand):
            return self._data.get(op.key)
        if isinstance(op, SetCommand):
            self._data[op.key] = op.value
            return None
        if isinstance(op, DeleteCommand):
            self._data.pop(op.key, None)
            return None
        if isinstance(op, CompareAndSwapCommand):
            current = self._data.get(op.key)
            if current != op.expected:
                return False
            self._data[op.key] = op.new_value
            return True
        raise TypeError(f"unknown command op: {op!r}")  # pragma: no cover

    def get(self, key: str) -> str | None:
        """Direct read of the current value. Note that reading a
        follower's state machine this way is not linearizable on its own
        (the follower may be lagging) -- see `raftkv.kv.client.KVClient`,
        which always routes reads through the leader the same as writes."""
        return self._data.get(key)

    def snapshot(self) -> dict[str, str]:
        """A point-in-time copy of the whole key space, for tests and for
        `RaftNode`-external snapshotting (not wired into the consensus
        layer itself -- see README for what raftkv does and doesn't
        implement)."""
        return dict(self._data)
