"""
End-to-end tests of `raftkv.transport.tcp.RaftServer` over real
sockets on localhost -- the counterpart to `test_election.py` /
`test_log_replication.py`'s use of `SimulatedCluster`, proving the same
`RaftNode` core actually works as a real multi-process (here,
multi-asyncio-server-within-one-process) system, not just in the
deterministic simulator.

These are the slowest tests in the suite (real wall-clock ticks, real
TCP handshakes) but still fast in absolute terms since the tick
interval is turned down to a few milliseconds -- see `_TICK_INTERVAL`.

`KVClient` itself is synchronous (plain blocking sockets, see
`tcp_sender`), so calling it directly from inside the asyncio event
loop that's also running the `RaftServer`s under test would block that
same loop and deadlock -- every client call below goes through
`asyncio.to_thread` for exactly that reason.
"""

import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from raftkv.kv.client import KVClient
from raftkv.transport.tcp import RaftServer, tcp_sender

_TICK_INTERVAL = 0.01
_BASE_PORT = 19000  # arbitrary high port range, offset per test to avoid collisions


class _Cluster:
    """Spins up N real RaftServer processes-in-one-event-loop on
    localhost, all talking real TCP, and tears them down cleanly."""

    def __init__(self, node_ids: list[int], port_offset: int):
        self.node_ids = node_ids
        self.peer_ports = {n: _BASE_PORT + port_offset + n for n in node_ids}
        self.client_ports = {n: _BASE_PORT + port_offset + 100 + n for n in node_ids}
        self.servers: dict[int, RaftServer] = {}

    async def start(self):
        all_peer_addrs = {n: ("127.0.0.1", self.peer_ports[n]) for n in self.node_ids}
        for n in self.node_ids:
            peers = {j: addr for j, addr in all_peer_addrs.items() if j != n}
            server = RaftServer(
                n,
                peers,
                peer_host="127.0.0.1",
                peer_port=self.peer_ports[n],
                client_host="127.0.0.1",
                client_port=self.client_ports[n],
                tick_interval=_TICK_INTERVAL,
            )
            self.servers[n] = server
            await server.start()

    async def stop(self):
        for server in list(self.servers.values()):
            await server.stop()

    def node_addrs(self) -> dict[int, tuple[str, int]]:
        return {n: ("127.0.0.1", self.client_ports[n]) for n in self.node_ids}

    async def wait_for_leader(self, timeout_seconds: float = 5.0) -> int:
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout_seconds
        while loop.time() < deadline:
            leaders = [n for n, s in self.servers.items() if s.node.role.value == "leader"]
            if leaders:
                return leaders[0]
            await asyncio.sleep(_TICK_INTERVAL)
        raise AssertionError("no leader elected over real TCP within timeout")

    def client(self) -> KVClient:
        send = tcp_sender(self.node_addrs(), timeout=2.0)
        return KVClient(self.node_ids, send)


class TestRealTcpCluster(unittest.IsolatedAsyncioTestCase):
    async def test_three_node_cluster_elects_a_leader_over_real_sockets(self):
        cluster = _Cluster([1, 2, 3], port_offset=0)
        await cluster.start()
        try:
            leader = await cluster.wait_for_leader()
            self.assertIn(leader, cluster.node_ids)
        finally:
            await cluster.stop()

    async def test_set_and_get_round_trip_over_real_sockets(self):
        cluster = _Cluster([1, 2, 3], port_offset=10)
        await cluster.start()
        try:
            await cluster.wait_for_leader()
            client = cluster.client()
            await asyncio.to_thread(client.set, "foo", "bar")
            value = await asyncio.to_thread(client.get, "foo")
            self.assertEqual(value, "bar")
        finally:
            await cluster.stop()

    async def test_cas_over_real_sockets(self):
        cluster = _Cluster([1, 2, 3], port_offset=20)
        await cluster.start()
        try:
            await cluster.wait_for_leader()
            client = cluster.client()
            await asyncio.to_thread(client.set, "x", "1")
            ok = await asyncio.to_thread(client.compare_and_swap, "x", "1", "2")
            self.assertTrue(ok)
            self.assertEqual(await asyncio.to_thread(client.get, "x"), "2")
            stale_ok = await asyncio.to_thread(client.compare_and_swap, "x", "1", "3")
            self.assertFalse(stale_ok)
        finally:
            await cluster.stop()

    async def test_client_fails_over_when_the_leader_process_is_stopped(self):
        cluster = _Cluster([1, 2, 3], port_offset=30)
        await cluster.start()
        try:
            old_leader = await cluster.wait_for_leader()
            client = cluster.client()
            await asyncio.to_thread(client.set, "before", "1")

            await cluster.servers[old_leader].stop()
            del cluster.servers[old_leader]

            await asyncio.to_thread(client.set, "after", "2")
            self.assertEqual(await asyncio.to_thread(client.get, "after"), "2")
            self.assertEqual(await asyncio.to_thread(client.get, "before"), "1")
        finally:
            await cluster.stop()

    async def test_delete_over_real_sockets(self):
        cluster = _Cluster([1, 2, 3], port_offset=50)
        await cluster.start()
        try:
            await cluster.wait_for_leader()
            client = cluster.client()
            await asyncio.to_thread(client.set, "x", "1")
            await asyncio.to_thread(client.delete, "x")
            self.assertIsNone(await asyncio.to_thread(client.get, "x"))
        finally:
            await cluster.stop()

    async def test_contacting_a_non_leader_node_first_still_succeeds(self):
        cluster = _Cluster([1, 2, 3], port_offset=40)
        await cluster.start()
        try:
            leader = await cluster.wait_for_leader()
            follower = next(n for n in cluster.node_ids if n != leader)
            client = cluster.client()
            client._known_leader = follower
            await asyncio.to_thread(client.set, "x", "1")
            self.assertEqual(await asyncio.to_thread(client.get, "x"), "1")
        finally:
            await cluster.stop()


if __name__ == "__main__":
    unittest.main()
