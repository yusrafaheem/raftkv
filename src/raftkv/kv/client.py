"""
raftkv.kv.client
===================

`KVClient`: a transport-agnostic client for a raftkv cluster.

Only the current leader accepts writes (and, in this project's
keep-it-simple design, reads too -- see `GetCommand`'s docstring in
`store.py`). A client that doesn't already know who the leader is has
to find out, and a client that's talking to the leader can lose that
status mid-conversation (the leader it's talking to gets partitioned
away, crashes, or steps down). Raft paper section 8 describes the
protocol this class implements: contact any server; a non-leader
rejects the request and, ideally, says who the leader is; remember the
last known leader and start there next time; on total silence, fall
back to trying servers round-robin.

This class knows nothing about sockets or the event loop -- it's driven
entirely through a caller-supplied `send(node_id, request) ->
ClientResponse | None` callable (`None` meaning "unreachable / timed
out"). `raftkv.transport.tcp` supplies a real one for the live demo
cluster; `raftkv.cluster.make_simulated_sender` supplies a synchronous
one over a `SimulatedCluster` for tests -- see
`tests/test_client.py` and `tests/test_linearizability.py`.
"""

from __future__ import annotations

import itertools
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from ..raft.types import NodeId
from .store import CompareAndSwapCommand, DeleteCommand, GetCommand, Op, SetCommand


@dataclass(frozen=True, slots=True)
class ClientRequest:
    client_id: str
    request_id: int
    op: Op


@dataclass(frozen=True, slots=True)
class ClientResponse:
    ok: bool
    result: object = None
    leader_hint: NodeId | None = None
    error: str | None = None


SendFn = Callable[[NodeId, ClientRequest], "ClientResponse | None"]


class ClientTimeoutError(Exception):
    """Raised when a request couldn't be completed within the client's
    retry budget -- e.g. the cluster has no leader (an election is still
    in progress, or fewer than a majority of nodes are reachable)."""


class KVClient:
    def __init__(
        self,
        node_ids: list[NodeId],
        send: SendFn,
        *,
        client_id: str | None = None,
        retry_delay: float = 0.05,
        max_retry_delay: float = 0.5,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not node_ids:
            raise ValueError("KVClient needs at least one node id to contact")
        self.node_ids = list(node_ids)
        self._send = send
        self.client_id = client_id or f"client-{uuid.uuid4().hex[:12]}"
        self._request_ids = itertools.count(1)
        self._known_leader: NodeId | None = None
        self._fallback_index = 0
        # A retry that lands on "not leader" or an unreachable node costs
        # essentially nothing in wall-clock time, but the cluster-side
        # event that would make the *next* retry succeed (a new election
        # completing) takes real time to happen. Retrying in a tight loop
        # would just burn through the attempt budget faster than the
        # cluster can possibly respond -- so failed attempts back off
        # (capped, doubling each time) before the next try. `sleep` is
        # injectable so tests can advance a fake clock instead of the
        # real one.
        self._retry_delay = retry_delay
        self._max_retry_delay = max_retry_delay
        self._sleep = sleep

    def get(self, key: str) -> str | None:
        return self._execute(GetCommand(key))

    def set(self, key: str, value: str) -> None:
        self._execute(SetCommand(key, value))

    def delete(self, key: str) -> None:
        self._execute(DeleteCommand(key))

    def compare_and_swap(self, key: str, expected: str | None, new_value: str) -> bool:
        return self._execute(CompareAndSwapCommand(key, expected, new_value))

    def _next_target(self) -> NodeId:
        if self._known_leader is not None:
            return self._known_leader
        return self.node_ids[self._fallback_index % len(self.node_ids)]

    def _rotate_fallback(self) -> None:
        self._known_leader = None
        self._fallback_index += 1

    def _execute(self, op: Op, *, max_attempts: int | None = None) -> object:
        # Reusing one ClientRequest (and therefore one request_id) across
        # every retry of this call is what makes retries idempotent --
        # see KVStateMachine's dedup docstring in store.py. Each *logical*
        # client call gets exactly one request_id, no matter how many
        # nodes it takes to actually land it.
        request = ClientRequest(self.client_id, next(self._request_ids), op)
        attempts = max_attempts if max_attempts is not None else 8 * len(self.node_ids) + 16
        last_error = "cluster unreachable"
        delay = self._retry_delay

        for attempt in range(attempts):
            target = self._next_target()
            response = self._send(target, request)

            if response is None:
                last_error = f"node {target} unreachable"
                self._rotate_fallback()
            elif response.ok:
                self._known_leader = target
                return response.result
            else:
                last_error = response.error or "request rejected"
                if response.leader_hint is not None:
                    self._known_leader = response.leader_hint
                else:
                    self._rotate_fallback()

            if attempt < attempts - 1:
                self._sleep(delay)
                delay = min(delay * 2, self._max_retry_delay)

        raise ClientTimeoutError(
            f"request {request.request_id} for client {self.client_id!r} did not "
            f"complete after {attempts} attempts (last error: {last_error})"
        )
