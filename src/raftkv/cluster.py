"""
raftkv.cluster
=================

`SimulatedCluster` wires together a set of `RaftNode`s, a
`SimulatedNetwork`, and one `KVStateMachine` per node into something a
test can drive one deterministic tick at a time. It is the thing
`tests/test_election.py`, `test_log_replication.py`, `test_safety.py`,
and `test_fault_tolerance.py` are all actually built on top of.

Each `tick()` does, in order:
  1. every live node's `tick()` (fires election timeouts / heartbeats),
  2. advance the network by one logical tick and hand delivered
     messages to their destination node's `step()`,
  3. for every live node, pull any newly-committed log entries and
     apply them to that node's own `KVStateMachine`.

"Killing" a node (`kill()`) just stops delivering it ticks and messages
-- its in-memory Raft state (log, term, voted_for) stays exactly as it
was, which models a process that crashes and is later restarted with
its persistent state intact (the durability guarantee real Raft
deployments get from fsync'd disk; raftkv's in-memory nodes get it for
free within a single simulated run, which is exactly what the
fault-tolerance tests need and no more).
"""

from __future__ import annotations

import random

from .kv.client import ClientRequest, ClientResponse
from .kv.store import Command, KVStateMachine
from .raft.node import DEFAULT_ELECTION_TIMEOUT_TICKS, DEFAULT_HEARTBEAT_INTERVAL_TICKS, RaftNode
from .raft.types import NodeId, Role
from .transport.simulated import SimulatedNetwork


class SimulatedCluster:
    def __init__(
        self,
        node_ids: list[NodeId],
        *,
        seed: int = 0,
        delay_range: tuple[int, int] = (1, 3),
        election_timeout_ticks: tuple[int, int] = DEFAULT_ELECTION_TIMEOUT_TICKS,
        heartbeat_interval_ticks: int = DEFAULT_HEARTBEAT_INTERVAL_TICKS,
    ) -> None:
        self.node_ids = list(node_ids)
        # random.Random() only supports None/int/float/str/bytes/bytearray
        # seeds without falling back to (deprecated) hash-based seeding, so
        # each RNG gets its own distinct *string* seed derived from the
        # cluster seed rather than a tuple.
        self.network = SimulatedNetwork(random.Random(f"{seed}:network"), delay_range=delay_range)
        self.nodes: dict[NodeId, RaftNode] = {
            node_id: RaftNode(
                node_id,
                node_ids,
                election_timeout_ticks=election_timeout_ticks,
                heartbeat_interval_ticks=heartbeat_interval_ticks,
                rng=random.Random(f"{seed}:node:{node_id}"),
            )
            for node_id in node_ids
        }
        self.state_machines: dict[NodeId, KVStateMachine] = {
            node_id: KVStateMachine() for node_id in node_ids
        }
        self.alive: set[NodeId] = set(node_ids)
        self.current_tick = 0

    def tick(self) -> None:
        self.current_tick += 1
        for node_id in self.alive:
            self.network.send_all(self.nodes[node_id].tick())

        for message in self.network.advance():
            if message.dst not in self.alive:
                continue
            self.network.send_all(self.nodes[message.dst].step(message))

        for node_id in self.alive:
            node = self.nodes[node_id]
            state_machine = self.state_machines[node_id]
            for entry in node.take_committed_entries():
                state_machine.apply(entry.command)

    def run(self, ticks: int) -> None:
        for _ in range(ticks):
            self.tick()

    def run_until(self, predicate, max_ticks: int = 1000) -> bool:
        """Tick until `predicate(self)` is true, up to `max_ticks`. Returns
        whether the predicate was satisfied (as opposed to timing out) --
        tests should assert on the return value rather than assuming
        success, so a broken invariant fails loudly instead of silently
        passing on a lucky partial run."""
        for _ in range(max_ticks):
            if predicate(self):
                return True
            self.tick()
        return predicate(self)

    # -- fault injection -----------------------------------------------------

    def kill(self, node_id: NodeId) -> None:
        self.alive.discard(node_id)

    def revive(self, node_id: NodeId) -> None:
        self.alive.add(node_id)

    def partition(self, groups: list[set[NodeId]]) -> None:
        self.network.partition(groups)

    def heal(self) -> None:
        self.network.heal()

    # -- inspection ------------------------------------------------------------

    def leaders(self) -> list[NodeId]:
        """Every live node that currently believes it's the leader. In a
        healthy cluster this has at most one element; election safety
        tests assert exactly that even across partitions and term
        changes (`tests/test_safety.py::test_at_most_one_leader_per_term`)."""
        return [n for n in self.alive if self.nodes[n].role is Role.LEADER]

    def leader(self) -> NodeId | None:
        current = self.leaders()
        return current[0] if len(current) == 1 else None

    def propose(self, command: object, *, via: NodeId | None = None):
        """Submit `command` to `via` (or the current unique leader, if
        `via` is omitted). Returns the `ProposeResult`, or `None` if
        there's no leader to propose to."""
        target = via if via is not None else self.leader()
        if target is None or target not in self.alive:
            return None
        result = self.nodes[target].propose(command)
        self.network.send_all(result.messages)
        return result

    def is_committed_everywhere(self, index: int) -> bool:
        return all(self.nodes[n].commit_index >= index for n in self.alive)


def make_simulated_sender(
    cluster: SimulatedCluster, *, max_wait_ticks: int = 300, settle_ticks_per_call: int = 5
):
    """Build a `KVClient`-compatible `send(node_id, request)` callable
    that drives a `SimulatedCluster` synchronously: proposes the request
    to `node_id` (if it's alive and currently leader), then ticks the
    cluster forward until the entry commits, gets overwritten (a
    conflicting leader change), or `max_wait_ticks` is exhausted.

    This plays the same role for tests that a real network round-trip
    plays for the live TCP demo -- it's what lets `tests/test_client.py`
    and `tests/test_linearizability.py` exercise `KVClient`'s retry and
    leader-discovery logic against a cluster that's actually electing
    leaders, replicating logs, and occasionally getting partitioned or
    losing nodes out from under the request.

    A real network call blocks for however long the round trip takes,
    during which the *cluster's* clock keeps advancing (elections time
    out, heartbeats fire) whether or not the client is watching. To
    model that, every call -- even one that's about to fail fast because
    `node_id` is dead or not the leader -- first ticks the cluster
    forward by `settle_ticks_per_call`. Without this, a client retry
    loop that only ever contacts dead or non-leader nodes would never
    give the cluster a chance to actually elect a new leader, since
    nothing else in the simulation is advancing time on its behalf.
    """

    def send(node_id, request: ClientRequest) -> ClientResponse | None:
        for _ in range(settle_ticks_per_call):
            cluster.tick()

        if node_id not in cluster.alive:
            return None
        node = cluster.nodes[node_id]
        if node.role is not Role.LEADER:
            return ClientResponse(ok=False, leader_hint=node.leader_id, error="not leader")

        command = Command(request.client_id, request.request_id, request.op)
        result = cluster.propose(command, via=node_id)
        if result is None or result.index is None:
            return ClientResponse(ok=False, leader_hint=node.leader_id, error="propose rejected")
        target_index, target_term = result.index, result.term

        for _ in range(max_wait_ticks):
            if node_id not in cluster.alive:
                return None
            current = cluster.nodes[node_id]
            if current.role is not Role.LEADER or current.current_term != target_term:
                return ClientResponse(
                    ok=False, leader_hint=current.leader_id, error="lost leadership before commit"
                )
            if current.log.term_at(target_index) != target_term:
                return ClientResponse(ok=False, error="entry overwritten before commit")
            if current.commit_index >= target_index:
                sm_result = cluster.state_machines[node_id].apply(command)
                return ClientResponse(ok=True, result=sm_result)
            cluster.tick()

        return None

    return send
