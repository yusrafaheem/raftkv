# Security

raftkv is a from-scratch learning/portfolio implementation of Raft consensus.
It has no authentication, encryption, or access control on either the peer
RPC port or the client port -- see the README's "Scope" section for the full
list of what this project deliberately does not implement.

**Do not use raftkv to store or transmit sensitive data, and do not expose
its ports beyond a trusted local network.** There is no supported security
posture for production use.

If you find a correctness bug in the consensus algorithm itself (as opposed
to a "missing hardening feature" already listed in Scope), please open an
issue with a seed and a minimal reproduction -- every test in this repo is
seeded and deterministic, so a failing run should reproduce exactly.
