"""
Benchmarks how many simulated ticks it takes a cluster to elect its
first leader, across cluster sizes and many seeds. This runs entirely
against `SimulatedCluster` -- no wall clock involved -- so the unit is
"ticks", not seconds, and the numbers are only meaningful relative to
each other (e.g. does a bigger cluster take longer to elect a leader?),
not as a prediction of real election latency for any particular
`tick_interval` a deployment might choose.

Run with: python3 benchmarks/bench_election_time.py
"""

from __future__ import annotations

import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from raftkv.cluster import SimulatedCluster  # noqa: E402


def ticks_to_first_leader(node_ids: list[int], seed: int, max_ticks: int = 1000) -> int:
    cluster = SimulatedCluster(node_ids, seed=seed)
    for tick in range(1, max_ticks + 1):
        cluster.tick()
        if cluster.leader() is not None:
            return tick
    raise RuntimeError(f"no leader elected within {max_ticks} ticks (seed={seed})")


def run(cluster_sizes: tuple[int, ...] = (3, 5, 7), num_seeds: int = 300) -> None:
    header = f"{'cluster size':>12} | {'min':>6} {'median':>8} {'mean':>8} {'p95':>8} {'max':>6}"
    print(f"ticks to first leader, {num_seeds} seeds per cluster size")
    print(header)
    print("-" * len(header))
    for size in cluster_sizes:
        node_ids = list(range(1, size + 1))
        samples = sorted(ticks_to_first_leader(node_ids, seed) for seed in range(num_seeds))
        p95 = samples[max(0, int(0.95 * len(samples)) - 1)]
        print(
            f"{size:>12} | {samples[0]:>6} {statistics.median(samples):>8.1f} "
            f"{statistics.mean(samples):>8.1f} {p95:>8} {samples[-1]:>6}"
        )


if __name__ == "__main__":
    run()
