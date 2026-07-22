# Changelog

## 0.1.0

Initial implementation:

- `RaftNode`: leader election, log replication, and the Figure-8 commit-safety
  rule, implemented as a pure state machine (`tick()` / `step()` / `propose()`).
- `SimulatedNetwork`: deterministic discrete-event transport with seeded
  delay/drop/partition injection, used by the entire test suite.
- `RaftServer`: a real asyncio TCP transport for a runnable multi-process
  cluster (`raftkv-node`, `raftkv-cli`).
- `KVStateMachine` and `KVClient`: a replicated key-value store on top of
  `RaftNode`, with linearizable reads (routed through the log) and
  client-request de-duplication (Raft paper section 8).
- 100+ tests: unit tests, randomized multi-seed safety-property checks,
  named fault-tolerance scenarios, and a linearizability chaos test.
