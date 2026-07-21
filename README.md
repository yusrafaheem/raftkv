# raftkv

A from-scratch implementation of the [Raft consensus algorithm](https://raft.github.io/raft.pdf) (Ongaro & Ousterhout, 2014) and a replicated key-value store built on top of it, in Python with no external dependencies.

This exists to demonstrate distributed-systems fundamentals end to end: leader election, log replication, the safety proofs Raft is actually known for, and the testing discipline needed to trust a consensus implementation at all. It's the fourth in a series of from-scratch systems projects (alongside [vectorgrad](../vectorgrad), an autodiff/ML engine, [ragent](../ragent), a RAG/agent infra project, and [minirel](../minirel), a relational database engine) -- this one specifically fills in distributed systems.

## Architecture

The core design choice: **consensus is a library, not a service.** `RaftNode` (`src/raftkv/raft/node.py`) is a pure state machine -- it performs no I/O, spawns no threads, and owns no clock. It exposes exactly three entry points:

- `tick()` -- advance one logical unit of time (fires election timeouts / heartbeats)
- `step(message)` -- process one incoming RPC (request or reply)
- `propose(command)` -- a client wants `command` appended to the log

Each returns the outbound messages the caller should deliver. What actually delivers them -- and what advances time -- is somebody else's problem entirely. This is the same architecture used by production implementations: etcd's `raft` package, TiKV's `raft-rs`, HashiCorp's `raft`. It's what makes exhaustive, deterministic testing possible: a test can drive thousands of ticks, inject dropped messages, kill a leader mid-write, and replay the exact same run byte-for-byte just by reusing a seed.

Two things drive the same `RaftNode` core:

- **`SimulatedNetwork`** (`src/raftkv/transport/simulated.py`) -- an in-process discrete-event simulator with controllable message delay, drops, and partitions, all seeded for reproducibility. No `time.sleep()`, no threads, no flakiness. This is what every test in this repo runs against.
- **`RaftServer`** (`src/raftkv/transport/tcp.py`) -- a real asyncio TCP service: one process per node, real sockets, a background tick loop on a real (if short) interval. This is what `raftkv-node` actually runs.

```
                 tick() / step() / propose()
                            |
                        RaftNode   <-- pure state machine, no I/O
                            |
              +-------------+-------------+
              |                           |
      SimulatedNetwork              RaftServer (asyncio TCP)
      (tests, deterministic)        (real cluster, real sockets)
```

On top of `RaftNode` sits `KVStateMachine` (`src/raftkv/kv/store.py`): a plain dict, mutated only by applying committed log entries in order. Reads (`GetCommand`) are routed through the log exactly like writes -- see that module's docstring for why: it's the simplest possible way to get linearizable reads, at the cost of paying a full replication round-trip per read (no ReadIndex/lease optimization here; see Scope below). `KVClient` (`src/raftkv/kv/client.py`) implements the client protocol from Raft paper section 8: find the leader, retry with backoff on rejection or timeout, and de-duplicate retried requests via a `(client_id, request_id)` tag so a retried non-idempotent command (a compare-and-swap, say) never gets applied twice.

## Running a real cluster

Three terminals:

```bash
pip install -e .

# terminal 1
raftkv-node --id 1 --peer 1=127.0.0.1:9001 --peer 2=127.0.0.1:9002 --peer 3=127.0.0.1:9003 --client-port 8001

# terminal 2
raftkv-node --id 2 --peer 1=127.0.0.1:9001 --peer 2=127.0.0.1:9002 --peer 3=127.0.0.1:9003 --client-port 8002

# terminal 3
raftkv-node --id 3 --peer 1=127.0.0.1:9001 --peer 2=127.0.0.1:9002 --peer 3=127.0.0.1:9003 --client-port 8003
```

Then, from a fourth terminal:

```bash
raftkv-cli --node 1=127.0.0.1:8001 --node 2=127.0.0.1:8002 --node 3=127.0.0.1:8003 set foo bar
raftkv-cli --node 1=127.0.0.1:8001 --node 2=127.0.0.1:8002 --node 3=127.0.0.1:8003 get foo
raftkv-cli --node 1=127.0.0.1:8001 --node 2=127.0.0.1:8002 --node 3=127.0.0.1:8003 cas foo bar baz
```

`raftkv-cli` doesn't need to know who the current leader is -- it'll contact any node, follow the leader-hint redirect if it guesses wrong, and retry with backoff if a node is unreachable. Kill whichever node you started as the leader (`Ctrl-C` in its terminal) and the next `set`/`get` call will transparently fail over to the new one once the remaining two nodes elect it.

## Testing

102 tests, `python -m unittest discover -s tests`, no external dependencies, ~2 seconds wall-clock for the whole suite (most of that from the 5 tests that spin up real TCP servers on localhost).

- `test_log.py` -- unit tests of the replicated log itself (append ordering, truncation, term lookups).
- `test_election.py` -- leader election: single-node clusters, 3- and 5-node clusters across many seeds, and direct unit tests of every `RequestVote` branch (stale term, already-voted, log-too-short, higher-term step-down, stale/late replies ignored).
- `test_log_replication.py` -- log replication end to end, plus direct unit tests of the `AppendEntries` conflict-resolution logic (Raft section 5.3) and, critically, a test that pins down [Figure 8 of the Raft paper](https://raft.github.io/raft.pdf): a leader must never commit a prior-term entry just because it's replicated to a majority.
- `test_safety.py` -- Raft's five headline safety properties (election safety, leader append-only, log matching, leader completeness, state machine safety), each checked across dozens of random seeds with faults injected mid-run, not asserted once on a hand-picked scenario.
- `test_fault_tolerance.py` -- named scenarios: kill the leader mid-write (both before and after a write commits), network partitions (minority can't commit, majority keeps going, a healed partition reconverges), and a node that misses a long stretch of the log catching back up after revival.
- `test_kv_store.py` -- the state machine itself, including the client-request de-duplication that makes retries of a non-idempotent compare-and-swap safe.
- `test_client.py` -- `KVClient`'s leader discovery, failover, and retry-budget-exhaustion behavior.
- `test_linearizability.py` -- see below.
- `test_codec.py` / `test_tcp_transport.py` -- wire-format round-trips and end-to-end tests of the real asyncio TCP transport (real sockets on localhost, not simulated).

**On the linearizability test specifically:** because every operation (including reads) is linearized by its position in the committed log, this project's linearizability check can be unusually direct compared to a generic black-box history checker. Several client identities issue randomly interleaved get/set/delete/cas requests against a cluster that's simultaneously being partitioned and having nodes killed and revived; the test then replays the one true committed log order against a fresh reference state machine and confirms every client-observed result matches exactly what that replay produces. It's both a linearizability check and a chaos test in one.

All of this mirrors the testing philosophy from [minirel](../minirel) (this series' relational-database project): prefer a deterministic simulation of a hard-to-trigger failure over trying to actually trigger it with real timing, and verify against a reference model wherever one is available rather than hand-picking expected outputs.

## Benchmarks

```bash
python3 benchmarks/bench_election_time.py   # ticks-to-first-leader across cluster sizes, 300 seeds each
python3 benchmarks/bench_throughput.py      # real TCP cluster, sequential + concurrent write throughput
```

`bench_throughput.py` runs the actual asyncio TCP transport -- real sockets, a real (if short) tick interval -- but all nodes are separate `asyncio` servers inside one Python process on localhost. That's a development-machine, loopback-network number: no real network latency, one process bound by the GIL and a single event loop, not independent machines. Treat the output as "does the whole pipeline work end to end at a reasonable clip," not as production capacity planning.

## Scope

What this project deliberately does not implement, and why:

- **No persistence.** `current_term`, `voted_for`, and the log all live in memory only. A real Raft node must `fsync` this state before replying to any RPC so it survives a process crash; `RaftServer` doesn't, so a real crash (not the *simulated* crashes `SimulatedCluster.kill()` models) loses that node's state entirely. Simulated crashes work correctly because `SimulatedCluster` keeps a "dead" node's in-memory state intact and just stops delivering it ticks/messages -- which is exactly the guarantee real persistence is meant to provide.
- **No log compaction / snapshotting.** The log grows without bound. A production system needs periodic snapshots (Raft paper section 7) so old entries can be discarded and a lagging follower can be caught up with a snapshot transfer instead of the entire log.
- **No cluster membership changes.** The set of nodes is fixed at cluster creation; adding or removing a node safely (Raft paper section 6, joint consensus) isn't implemented.
- **Reads pay a full replication round-trip.** As noted above, this project routes reads through the log for simplicity and a straightforward linearizability argument, rather than implementing the ReadIndex or lease-based read-only optimization the paper describes.

What it does implement correctly and tests rigorously: leader election with randomized timeouts and split-vote resolution, log replication with the full AppendEntries consistency check and conflict resolution, the Figure-8 commit-safety rule, and all five of Raft's core safety properties -- verified deterministically across many seeds and fault-injection scenarios, not just demonstrated on a happy path.

## Prior art

- Diego Ongaro and John Ousterhout, ["In Search of an Understandable Consensus Algorithm"](https://raft.github.io/raft.pdf) (the Raft paper) -- the algorithm this project implements directly from, including the specific figure/section citations in the code's docstrings.
- [etcd's `raft` package](https://github.com/etcd-io/raft) and [TiKV's `raft-rs`](https://github.com/tikv/raft-rs) -- the "consensus as a pure library with tick/step/ready" architecture this project's `RaftNode` follows.
- The [MIT 6.824 (Distributed Systems)](https://pdos.csail.mit.edu/6.824/) lab structure, whose emphasis on deterministic, seed-reproducible testing over real-timing-dependent tests shaped this project's test suite.
