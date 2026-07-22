"""
raftkv's whole reason for routing reads through the log (see
`GetCommand`'s docstring in `raftkv/kv/store.py`) is to get
linearizability for free: every operation -- read or write -- is
linearized by its position in the committed log, so *the committed log
itself is the linearization order*. That makes this project's
linearizability check unusually direct compared to a generic black-box
observer (e.g. a Wing & Gong / Knossos-style checker that has to search
over possible orderings from just start/end timestamps): here, the test
can read the true global order straight out of the system under test
and simply confirm every client-visible result is exactly what replaying
that order against a reference sequential model would produce.

The check, precisely:

  1. Agreement -- every live replica's committed log holds the same
     command at the same index (Raft's state machine safety property,
     exercised on its own in test_safety.py; re-checked here as a
     precondition).
  2. Consistency -- for every request a simulated client got a
     definitive ("ok") response to, tagged with that request's own
     (client_id, request_id), replaying the committed log up through
     the exact entry carrying that tag against a fresh reference
     `KVStateMachine` reproduces exactly the result the client received.

Several distinct client identities issue randomly interleaved
Get/Set/Delete/Cas requests against a cluster that's simultaneously
being partitioned and having nodes killed/revived mid-stream -- so this
is a chaos test in its own right, not just a linearizability check
against an already-quiet cluster.
"""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from raftkv.cluster import SimulatedCluster, make_simulated_sender
from raftkv.kv.client import ClientRequest
from raftkv.kv.store import (
    CompareAndSwapCommand,
    DeleteCommand,
    GetCommand,
    KVStateMachine,
    SetCommand,
)


def _random_op(rng: random.Random, keys: list[str]):
    key = rng.choice(keys)
    kind = rng.choice(["get", "set", "delete", "cas"])
    if kind == "get":
        return GetCommand(key)
    if kind == "set":
        return SetCommand(key, f"v{rng.randint(0, 999)}")
    if kind == "delete":
        return DeleteCommand(key)
    return CompareAndSwapCommand(
        key, rng.choice([None, f"v{rng.randint(0, 999)}"]), f"v{rng.randint(0, 999)}"
    )


class _ChaosClient:
    """A minimal hand-rolled client (deliberately not `KVClient`) that
    tracks its own leader guess and retries, but -- unlike `KVClient` --
    exposes the exact `(client_id, request_id)` tag of every request it
    successfully completed, which is what lets this test match a
    client-observed result back to one specific committed log entry."""

    def __init__(self, client_id: str, node_ids: list[int], send, rng: random.Random):
        self.client_id = client_id
        self.node_ids = list(node_ids)
        self._send = send
        self._rng = rng
        self._next_request_id = 1
        self._known_leader = None

    def try_execute(self, op, max_attempts: int = 40):
        """Returns (request_id, result) on success, or None if the request
        never got a definitive answer within the attempt budget."""
        request_id = self._next_request_id
        self._next_request_id += 1
        request = ClientRequest(self.client_id, request_id, op)

        for _ in range(max_attempts):
            if self._known_leader is not None:
                target = self._known_leader
            else:
                target = self._rng.choice(self.node_ids)
            response = self._send(target, request)
            if response is None:
                self._known_leader = None
                continue
            if response.ok:
                self._known_leader = target
                return request_id, response.result
            self._known_leader = response.leader_hint
        return None


class TestLinearizabilityUnderChaos(unittest.TestCase):
    def _run_one(self, seed: int, num_clients: int = 4, num_ops: int = 80):
        c = SimulatedCluster([1, 2, 3, 4, 5], seed=seed)
        c.run_until(lambda cl: cl.leader() is not None, max_ticks=300)
        rng = random.Random(f"ops:{seed}")
        keys = ["a", "b", "c"]

        send = make_simulated_sender(c, max_wait_ticks=150)
        clients = [
            _ChaosClient(f"client-{i}", c.node_ids, send, random.Random(f"client:{seed}:{i}"))
            for i in range(num_clients)
        ]

        # (client_id, request_id) -> result the client actually received.
        observed: dict[tuple[str, int], object] = {}

        for _ in range(num_ops):
            client = rng.choice(clients)
            op = _random_op(rng, keys)
            outcome = client.try_execute(op)
            if outcome is not None:
                request_id, result = outcome
                observed[(client.client_id, request_id)] = result

            if rng.random() < 0.12 and len(c.alive) > 3:
                c.kill(rng.choice(list(c.alive)))
            if rng.random() < 0.08:
                dead = set(c.node_ids) - c.alive
                if dead:
                    c.revive(rng.choice(list(dead)))
            if rng.random() < 0.05:
                c.heal()

        c.heal()
        for n in set(c.node_ids) - c.alive:
            c.revive(n)
        c.run(300)  # let everything quiesce and fully replicate

        return c, observed

    def _committed_log_agrees_across_replicas(self, c: SimulatedCluster) -> int:
        agreed_index = min(c.nodes[n].commit_index for n in c.node_ids)
        for idx in range(1, agreed_index + 1):
            commands = {c.nodes[n].log.get(idx).command for n in c.node_ids}
            self.assertEqual(len(commands), 1, f"replicas disagree on committed index {idx}")
        return agreed_index

    def _check_every_observed_result_matches_the_replay(
        self, c: SimulatedCluster, observed: dict, agreed_index: int
    ) -> None:
        reference = KVStateMachine()
        node = c.nodes[c.node_ids[0]]
        expected_by_tag: dict[tuple[str, int], object] = {}
        for idx in range(1, agreed_index + 1):
            entry = node.log.get(idx)
            command = entry.command
            result = reference.apply(command)
            expected_by_tag[(command.client_id, command.request_id)] = result

        mismatches = []
        for tag, client_result in observed.items():
            if tag not in expected_by_tag:
                mismatches.append((tag, "no committed entry found", client_result))
                continue
            expected = expected_by_tag[tag]
            if expected != client_result:
                mismatches.append((tag, expected, client_result))

        self.assertEqual(
            mismatches,
            [],
            f"{len(mismatches)} client-observed results diverged from the replayed "
            f"committed log -- (tag, expected, client-saw): {mismatches[:5]}",
        )

    def test_linearizable_history_across_several_seeds_under_chaos(self):
        for seed in range(15):
            with self.subTest(seed=seed):
                c, observed = self._run_one(seed)
                self.assertGreater(len(observed), 0, "no requests completed -- test is vacuous")
                agreed_index = self._committed_log_agrees_across_replicas(c)
                self._check_every_observed_result_matches_the_replay(c, observed, agreed_index)


if __name__ == "__main__":
    unittest.main()
