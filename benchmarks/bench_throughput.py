"""
Measures real wall-clock write throughput and latency of a raftkv
cluster running the actual asyncio TCP transport -- multiple
`RaftServer` instances, real sockets, real event-loop scheduling (see
`raftkv/transport/tcp.py`) -- all within a single process on localhost.

This is a development-machine, loopback-network number: no real
network latency, one Python process bound by the GIL and a single
asyncio event loop, not independent machines on a real network. Treat
it as "does the whole pipeline actually work end-to-end at a reasonable
clip", not as capacity planning for a production deployment -- see the
README's benchmark section for that caveat spelled out in full.

Run with: python3 benchmarks/bench_throughput.py
"""

from __future__ import annotations

import asyncio
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from raftkv.kv.client import KVClient  # noqa: E402
from raftkv.transport.tcp import RaftServer, tcp_sender  # noqa: E402

BASE_PORT = 22000
TICK_INTERVAL = 0.01


async def start_cluster(node_ids: list[int]):
    peer_ports = {n: BASE_PORT + n for n in node_ids}
    client_ports = {n: BASE_PORT + 100 + n for n in node_ids}
    all_peer_addrs = {n: ("127.0.0.1", peer_ports[n]) for n in node_ids}
    servers: dict[int, RaftServer] = {}
    for n in node_ids:
        peers = {j: addr for j, addr in all_peer_addrs.items() if j != n}
        server = RaftServer(
            n,
            peers,
            peer_host="127.0.0.1",
            peer_port=peer_ports[n],
            client_host="127.0.0.1",
            client_port=client_ports[n],
            tick_interval=TICK_INTERVAL,
        )
        servers[n] = server
        await server.start()
    return servers, client_ports


async def wait_for_leader(servers: dict[int, RaftServer], timeout_seconds: float = 5.0) -> int:
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout_seconds
    while loop.time() < deadline:
        leaders = [n for n, s in servers.items() if s.node.role.value == "leader"]
        if leaders:
            return leaders[0]
        await asyncio.sleep(TICK_INTERVAL)
    raise RuntimeError("no leader elected within timeout")


async def bench_sequential_writes(client: KVClient, n: int) -> list[float]:
    """Each write waits for the previous one to fully commit before the
    next is issued -- this measures per-operation latency, not how many
    operations the cluster can have in flight at once."""
    latencies = []
    for i in range(n):
        start = time.perf_counter()
        await asyncio.to_thread(client.set, f"k{i % 50}", f"v{i}")
        latencies.append(time.perf_counter() - start)
    return latencies


async def bench_concurrent_writes(client_pool: list[KVClient], ops_per_client: int) -> float:
    """Several independent clients firing writes concurrently (via
    separate threads, since KVClient itself is blocking) -- measures
    aggregate throughput under concurrent load rather than one
    operation at a time."""

    async def worker(client: KVClient, offset: int):
        for i in range(ops_per_client):
            await asyncio.to_thread(client.set, f"k{(offset + i) % 50}", f"v{i}")

    start = time.perf_counter()
    await asyncio.gather(*(worker(c, i * ops_per_client) for i, c in enumerate(client_pool)))
    return time.perf_counter() - start


def report_latencies(name: str, latencies: list[float], wall_seconds: float) -> None:
    ms = sorted(v * 1000 for v in latencies)
    p50 = ms[len(ms) // 2]
    p95 = ms[max(0, int(len(ms) * 0.95) - 1)]
    ops_per_sec = len(ms) / wall_seconds
    print(f"{name}: {len(ms)} ops in {wall_seconds:.2f}s -> {ops_per_sec:.1f} ops/sec")
    print(
        f"  latency (ms): mean={statistics.mean(ms):.2f} p50={p50:.2f} "
        f"p95={p95:.2f} max={ms[-1]:.2f}"
    )


async def run_for_cluster_size(size: int) -> None:
    node_ids = list(range(1, size + 1))
    servers, client_ports = await start_cluster(node_ids)
    try:
        await wait_for_leader(servers)
        node_addrs = {n: ("127.0.0.1", client_ports[n]) for n in node_ids}

        client = KVClient(node_ids, tcp_sender(node_addrs))
        start = time.perf_counter()
        latencies = await bench_sequential_writes(client, n=150)
        wall = time.perf_counter() - start
        report_latencies(f"{size}-node cluster, sequential SET", latencies, wall)

        pool = [
            KVClient(node_ids, tcp_sender(node_addrs), client_id=f"bench-{i}") for i in range(8)
        ]
        wall = await bench_concurrent_writes(pool, ops_per_client=25)
        total_ops = 8 * 25
        print(
            f"{size}-node cluster, 8 concurrent clients x 25 SETs: "
            f"{total_ops} ops in {wall:.2f}s -> {total_ops / wall:.1f} ops/sec"
        )
    finally:
        for server in servers.values():
            await server.stop()
    print()


async def main() -> None:
    for size in (3, 5):
        await run_for_cluster_size(size)


if __name__ == "__main__":
    asyncio.run(main())
