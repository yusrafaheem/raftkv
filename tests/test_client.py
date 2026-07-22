import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from raftkv.cluster import SimulatedCluster, make_simulated_sender
from raftkv.kv.client import ClientResponse, ClientTimeoutError, KVClient


def elected_cluster(node_ids=(1, 2, 3), seed=0):
    c = SimulatedCluster(list(node_ids), seed=seed)
    ok = c.run_until(lambda cl: cl.leader() is not None, max_ticks=300)
    assert ok
    return c


def make_client(cluster, **kwargs):
    send = make_simulated_sender(cluster)
    return KVClient(cluster.node_ids, send, sleep=lambda _: None, **kwargs)


class TestBasicClientOperations(unittest.TestCase):
    def test_set_and_get_round_trip(self):
        c = elected_cluster(seed=1)
        client = make_client(c)
        client.set("x", "1")
        self.assertEqual(client.get("x"), "1")

    def test_get_of_missing_key_is_none(self):
        c = elected_cluster(seed=2)
        client = make_client(c)
        self.assertIsNone(client.get("missing"))

    def test_delete_removes_the_key(self):
        c = elected_cluster(seed=3)
        client = make_client(c)
        client.set("x", "1")
        client.delete("x")
        self.assertIsNone(client.get("x"))

    def test_compare_and_swap_success_and_failure(self):
        c = elected_cluster(seed=4)
        client = make_client(c)
        client.set("x", "1")
        self.assertTrue(client.compare_and_swap("x", "1", "2"))
        self.assertEqual(client.get("x"), "2")
        self.assertFalse(client.compare_and_swap("x", "stale", "3"))
        self.assertEqual(client.get("x"), "2")

    def test_two_independently_constructed_clients_get_distinct_generated_ids(self):
        c = elected_cluster(seed=21)
        first = make_client(c)
        second = make_client(c)
        self.assertNotEqual(first.client_id, second.client_id)

    def test_client_starts_with_no_known_leader_and_still_succeeds(self):
        c = elected_cluster(seed=5)
        client = make_client(c)
        self.assertIsNone(client._known_leader)
        client.set("x", "1")
        self.assertIsNotNone(client._known_leader)  # learned it from the first request


class TestLeaderDiscoveryAndFailover(unittest.TestCase):
    def test_contacting_a_follower_first_still_succeeds_via_the_leader_hint(self):
        c = elected_cluster(seed=6)
        follower = next(n for n in c.node_ids if n != c.leader())
        client = make_client(c)
        client._known_leader = follower  # force the client to guess wrong first
        client.set("x", "1")
        self.assertEqual(client.get("x"), "1")

    def test_client_transparently_survives_a_leader_crash(self):
        c = elected_cluster(seed=7)
        client = make_client(c)
        client.set("x", "1")
        old_leader = c.leader()
        c.kill(old_leader)
        client.set("y", "2")  # must retry through leader-hint/fallback + real election
        self.assertEqual(client.get("y"), "2")
        self.assertNotEqual(c.leader(), old_leader)

    def test_client_raises_a_clear_error_when_no_majority_is_available(self):
        c = elected_cluster(node_ids=(1, 2, 3), seed=8)
        client = make_client(c)
        # kill 2 of 3 nodes -- no majority can ever be reached
        alive = list(c.alive)
        for n in alive[:2]:
            c.kill(n)
        with self.assertRaises(ClientTimeoutError):
            client.set("x", "1")


class TestRequestIdempotencyAcrossRetries(unittest.TestCase):
    def test_a_manually_retried_send_never_double_applies(self):
        # Bypass KVClient entirely and drive the raw `send` function
        # directly with the exact same ClientRequest twice, simulating a
        # client that never saw the first response and retried -- this
        # isolates the dedup behavior itself from KVClient's own
        # request-id bookkeeping (which is exercised separately above).
        c = elected_cluster(seed=9)
        send = make_simulated_sender(c)

        from raftkv.kv.client import ClientRequest
        from raftkv.kv.store import CompareAndSwapCommand

        leader = c.leader()
        req = ClientRequest("manual-client", 1, CompareAndSwapCommand("counter", None, "once"))
        first = send(leader, req)
        second = send(leader, req)  # exact retry of the identical request
        self.assertTrue(first.ok)
        self.assertTrue(second.ok)
        self.assertEqual(first.result, second.result)
        self.assertEqual(c.state_machines[c.leader()].get("counter"), "once")


class TestResponseShape(unittest.TestCase):
    def test_client_response_defaults(self):
        r = ClientResponse(ok=True)
        self.assertIsNone(r.result)
        self.assertIsNone(r.leader_hint)
        self.assertIsNone(r.error)


class TestConstruction(unittest.TestCase):
    def test_empty_node_ids_list_is_rejected(self):
        with self.assertRaises(ValueError):
            KVClient([], send=lambda node_id, request: None)

    def test_a_caller_supplied_client_id_is_used_instead_of_a_generated_one(self):
        c = elected_cluster(seed=20)
        client = make_client(c, client_id="fixed-id")
        self.assertEqual(client.client_id, "fixed-id")


if __name__ == "__main__":
    unittest.main()
