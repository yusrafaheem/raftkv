"""
raftkv.raft.node
===================

`RaftNode` is the whole consensus algorithm (Raft paper sections 5.1-5.4:
leader election, log replication, and safety) implemented as a pure
state machine. It performs no I/O, spawns no threads, and owns no
clock. Every entry point returns the outbound messages the caller
should deliver -- what actually delivers them (a deterministic
in-process simulator for tests, or a real network transport for the
runnable demo) is entirely someone else's concern.

Three entry points cover everything:

  tick()             advance one logical time step; fires election
                      timeouts (follower/candidate) or heartbeats (leader).
  step(message)       process one incoming RPC request or reply.
  propose(command)    client wants to append `command`; only takes effect
                      if this node is currently the leader.

This split (a side-effect-free core plus an external driver loop) is
the same architecture used by etcd's raft package and TiKV's raft-rs --
it's what makes it possible to write deterministic tests that simulate
thousands of ticks, dropped messages, and leader crashes without a
single `time.sleep()` or thread (see `tests/test_safety.py` and
`tests/test_fault_tolerance.py`).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .log import LogEntry, RaftLog
from .rpc import (
    AppendEntriesArgs,
    AppendEntriesReply,
    Message,
    RequestVoteArgs,
    RequestVoteReply,
)
from .types import LogIndex, NodeId, Role, Term

DEFAULT_ELECTION_TIMEOUT_TICKS = (10, 20)
DEFAULT_HEARTBEAT_INTERVAL_TICKS = 3


@dataclass(frozen=True, slots=True)
class ProposeResult:
    """Outcome of a `propose()` call.

    `index` is `None` when the node isn't currently the leader and the
    command was rejected outright -- the caller (typically a KV client
    sitting behind a leader-redirect loop) should treat that as "try a
    different node". When `index` is set, the entry has been appended to
    *this* node's log and replication has started, but it is not yet
    committed: the caller must still wait for `commit_index` to reach
    `index` (and for that entry to still be present at that index --
    see the module docstring on step-down/overwrite races) before
    treating the command as durable.
    """

    index: LogIndex | None
    term: Term
    messages: tuple[Message, ...]


def _majority(peer_count: int) -> int:
    """Smallest vote/ack count that constitutes a majority of a cluster of
    `peer_count + 1` nodes (peers plus self)."""
    return (peer_count + 1) // 2 + 1


class RaftNode:
    def __init__(
        self,
        node_id: NodeId,
        peer_ids: list[NodeId] | tuple[NodeId, ...],
        *,
        election_timeout_ticks: tuple[int, int] = DEFAULT_ELECTION_TIMEOUT_TICKS,
        heartbeat_interval_ticks: int = DEFAULT_HEARTBEAT_INTERVAL_TICKS,
        rng: random.Random | None = None,
    ) -> None:
        self.id = node_id
        self.peer_ids: tuple[NodeId, ...] = tuple(p for p in peer_ids if p != node_id)

        # Persistent state (Figure 2 of the Raft paper). A real deployment
        # would fsync these to disk before replying to any RPC; raftkv
        # keeps them in memory only -- see the README's "what this project
        # deliberately does not do" section.
        self.current_term: Term = 0
        self.voted_for: NodeId | None = None
        self.log = RaftLog()

        # Volatile state, all nodes.
        self.commit_index: LogIndex = 0
        self.last_applied: LogIndex = 0

        # Volatile state, leaders only (reinitialized on every election win).
        self.next_index: dict[NodeId, LogIndex] = {}
        self.match_index: dict[NodeId, LogIndex] = {}

        self.role: Role = Role.FOLLOWER
        self.leader_id: NodeId | None = None
        self.votes_received: set[NodeId] = set()

        self._election_timeout_range = election_timeout_ticks
        self._heartbeat_interval = heartbeat_interval_ticks
        self._ticks_since_reset = 0
        self._ticks_since_heartbeat = 0
        self._rng = rng if rng is not None else random.Random(node_id)
        self._election_deadline = self._rng.randint(*election_timeout_ticks)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return (
            f"RaftNode(id={self.id}, role={self.role.value}, term={self.current_term}, "
            f"log_len={self.log.last_index}, commit={self.commit_index})"
        )

    # -- driver-facing entry points --------------------------------------

    def tick(self) -> list[Message]:
        """Advance one logical unit of time.

        Followers and candidates count ticks toward a randomized election
        timeout; when it elapses, they (re)start an election. Leaders
        count ticks toward a fixed heartbeat interval and broadcast an
        (possibly-empty) AppendEntries when it elapses -- this is both
        the heartbeat that suppresses followers' election timers and the
        mechanism that eventually retries replication to lagging peers.
        """
        if self.role is Role.LEADER:
            self._ticks_since_heartbeat += 1
            if self._ticks_since_heartbeat >= self._heartbeat_interval:
                self._ticks_since_heartbeat = 0
                return self._broadcast_append_entries()
            return []

        self._ticks_since_reset += 1
        if self._ticks_since_reset >= self._election_deadline:
            return self._start_election()
        return []

    def propose(self, command: object) -> ProposeResult:
        """Client wants `command` appended to the replicated log. Only the
        current leader can accept proposals -- see `ProposeResult`'s
        docstring for how a caller should handle rejection."""
        if self.role is not Role.LEADER:
            return ProposeResult(index=None, term=self.current_term, messages=())
        entry = LogEntry(term=self.current_term, index=self.log.last_index + 1, command=command)
        self.log.append(entry)
        self.match_index[self.id] = self.log.last_index
        # Replicate immediately rather than waiting for the next heartbeat
        # tick -- this is what keeps write latency bounded by network RTT
        # instead of by the heartbeat interval.
        self._ticks_since_heartbeat = 0
        messages = self._broadcast_append_entries()
        return ProposeResult(index=entry.index, term=self.current_term, messages=tuple(messages))

    def step(self, message: Message) -> list[Message]:
        """Process one incoming RPC (request or reply) and return whatever
        outbound messages it produces."""
        payload = message.payload
        if isinstance(payload, RequestVoteArgs):
            return self._on_request_vote(message.src, payload)
        if isinstance(payload, RequestVoteReply):
            return self._on_request_vote_reply(message.src, payload)
        if isinstance(payload, AppendEntriesArgs):
            return self._on_append_entries(message.src, payload)
        if isinstance(payload, AppendEntriesReply):
            return self._on_append_entries_reply(message.src, payload)
        raise TypeError(f"unrecognized RPC payload: {payload!r}")  # pragma: no cover

    def take_committed_entries(self) -> list[LogEntry]:
        """Return entries newly committed since the last call (i.e. with
        index in `(last_applied, commit_index]`) and advance
        `last_applied` past them. The caller is responsible for actually
        applying each entry's command to a state machine, in order --
        this method only manages the Raft-level bookkeeping of which
        entries have been handed off."""
        if self.commit_index <= self.last_applied:
            return []
        entries = self.log.entries_between(self.last_applied + 1, self.commit_index)
        self.last_applied = self.commit_index
        return entries

    # -- election ----------------------------------------------------------

    def _reset_election_timer(self) -> None:
        self._ticks_since_reset = 0
        self._election_deadline = self._rng.randint(*self._election_timeout_range)

    def _become_follower(self, term: Term) -> None:
        """Step down to follower, adopting `term` if it's newer than ours.
        Called whenever we observe an RPC (request or reply) carrying a
        term higher than `current_term` -- Raft's "if RPC request or
        response contains term T > currentTerm: set currentTerm = T,
        convert to follower" rule (Figure 2)."""
        if term > self.current_term:
            self.current_term = term
            self.voted_for = None
        self.role = Role.FOLLOWER
        self.leader_id = None
        self._reset_election_timer()

    def _start_election(self) -> list[Message]:
        self.role = Role.CANDIDATE
        self.current_term += 1
        self.voted_for = self.id
        self.votes_received = {self.id}
        self.leader_id = None
        self._reset_election_timer()

        if not self.peer_ids:
            # Single-node "cluster": we trivially have a majority of one.
            return self._become_leader()

        return [
            Message(
                self.id,
                peer,
                RequestVoteArgs(
                    term=self.current_term,
                    candidate_id=self.id,
                    last_log_index=self.log.last_index,
                    last_log_term=self.log.last_term(),
                ),
            )
            for peer in self.peer_ids
        ]

    def _become_leader(self) -> list[Message]:
        self.role = Role.LEADER
        self.leader_id = self.id
        self.next_index = {p: self.log.last_index + 1 for p in self.peer_ids}
        self.match_index = {p: 0 for p in self.peer_ids}
        self.match_index[self.id] = self.log.last_index
        self._ticks_since_heartbeat = 0
        # Send an immediate heartbeat rather than waiting for the next
        # tick, so followers learn about the new leader (and stop their
        # own election timers) as soon as possible.
        return self._broadcast_append_entries()

    def _on_request_vote(self, src: NodeId, args: RequestVoteArgs) -> list[Message]:
        if args.term > self.current_term:
            self._become_follower(args.term)

        if args.term < self.current_term:
            return [self._vote_reply(src, granted=False)]

        log_is_at_least_as_up_to_date = args.last_log_term > self.log.last_term() or (
            args.last_log_term == self.log.last_term()
            and args.last_log_index >= self.log.last_index
        )
        already_voted_for_someone_else = self.voted_for not in (None, args.candidate_id)

        if already_voted_for_someone_else or not log_is_at_least_as_up_to_date:
            return [self._vote_reply(src, granted=False)]

        self.voted_for = args.candidate_id
        self._reset_election_timer()
        return [self._vote_reply(src, granted=True)]

    def _vote_reply(self, dst: NodeId, *, granted: bool) -> Message:
        return Message(self.id, dst, RequestVoteReply(self.current_term, granted, self.id))

    def _on_request_vote_reply(self, src: NodeId, reply: RequestVoteReply) -> list[Message]:
        if reply.term > self.current_term:
            self._become_follower(reply.term)
            return []
        if self.role is not Role.CANDIDATE or reply.term != self.current_term:
            return []  # stale reply from an earlier term, or we're no longer a candidate

        if reply.vote_granted:
            self.votes_received.add(reply.voter_id)
            if len(self.votes_received) >= _majority(len(self.peer_ids)):
                return self._become_leader()
        return []

    # -- log replication -----------------------------------------------------

    def _broadcast_append_entries(self) -> list[Message]:
        return [self._append_entries_for(peer) for peer in self.peer_ids]

    def _append_entries_for(self, peer: NodeId) -> Message:
        next_idx = self.next_index.get(peer, self.log.last_index + 1)
        prev_index = next_idx - 1
        prev_term = self.log.term_at(prev_index)
        entries = tuple(self.log.entries_from(next_idx))
        return Message(
            self.id,
            peer,
            AppendEntriesArgs(
                term=self.current_term,
                leader_id=self.id,
                prev_log_index=prev_index,
                prev_log_term=prev_term,
                entries=entries,
                leader_commit=self.commit_index,
            ),
        )

    def _on_append_entries(self, src: NodeId, args: AppendEntriesArgs) -> list[Message]:
        if args.term > self.current_term:
            self._become_follower(args.term)

        if args.term < self.current_term:
            return [self._append_reply(src, success=False, match_index=0)]

        # A valid AppendEntries for our current (or newly-adopted) term
        # means `src` is the legitimate leader: recognize it, and if we
        # were a candidate for this term, stand down (Figure 2's "while
        # waiting for votes, candidate may receive AppendEntries RPC from
        # another server claiming to be leader... if the leader's term is
        # at least as large as the candidate's, the candidate recognizes
        # the leader as legitimate and returns to follower state").
        self.leader_id = args.leader_id
        self.role = Role.FOLLOWER
        self._reset_election_timer()

        if args.prev_log_index > 0 and self.log.term_at(args.prev_log_index) != args.prev_log_term:
            return [self._append_reply(src, success=False, match_index=0)]

        for entry in args.entries:
            existing_term = self.log.term_at(entry.index)
            if existing_term != 0 and existing_term != entry.term:
                # Conflict: an existing entry at this index disagrees with
                # the leader. Delete it and everything after it (section
                # 5.3), then fall through to append the leader's version.
                self.log.truncate_from(entry.index)
                existing_term = 0
            if existing_term == 0:
                self.log.append(entry)
            # else: we already have this exact entry (same term) -- a
            # duplicate/retried RPC, nothing to do.

        if args.leader_commit > self.commit_index:
            self.commit_index = min(args.leader_commit, self.log.last_index)

        matched = args.prev_log_index + len(args.entries)
        return [self._append_reply(src, success=True, match_index=matched)]

    def _append_reply(self, dst: NodeId, *, success: bool, match_index: LogIndex) -> Message:
        return Message(
            self.id, dst, AppendEntriesReply(self.current_term, success, self.id, match_index)
        )

    def _on_append_entries_reply(self, src: NodeId, reply: AppendEntriesReply) -> list[Message]:
        if reply.term > self.current_term:
            self._become_follower(reply.term)
            return []
        if self.role is not Role.LEADER or reply.term != self.current_term:
            return []  # stale reply, or we're no longer leader for this term

        if reply.success:
            self.match_index[src] = max(self.match_index.get(src, 0), reply.match_index)
            self.next_index[src] = self.match_index[src] + 1
            self._advance_commit_index()
            return []

        # Log inconsistency: back off `next_index` and immediately retry
        # rather than waiting for the next heartbeat tick, so a lagging
        # follower catches up in O(conflicting entries) round trips
        # instead of O(conflicting entries * heartbeat interval).
        self.next_index[src] = max(1, self.next_index.get(src, 1) - 1)
        return [self._append_entries_for(src)]

    def _advance_commit_index(self) -> None:
        """Raft section 5.3/5.4: a leader commits index N once a majority
        of `match_index` values are >= N *and* the entry at N was created
        in the leader's current term -- committing an entry from an
        earlier term just because it's now replicated to a majority is
        exactly the unsafe case Figure 8 of the paper warns about, so the
        term check here is not optional.
        """
        for n in range(self.log.last_index, self.commit_index, -1):
            if self.log.term_at(n) != self.current_term:
                continue
            replicated_count = 1  # ourselves
            replicated_count += sum(1 for p in self.peer_ids if self.match_index.get(p, 0) >= n)
            if replicated_count >= _majority(len(self.peer_ids)):
                self.commit_index = n
                return
