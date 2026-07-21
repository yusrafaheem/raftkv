"""
raftkv.raft.rpc
==================

The wire vocabulary of Raft: the two RPCs the paper defines
(RequestVote and AppendEntries), their replies, and a `Message`
envelope that pairs a payload with a sender/recipient so a transport
layer can route it without knowing anything about Raft semantics.

These are all frozen dataclasses -- once a message exists, nothing
should mutate it in flight. That matters most for `AppendEntriesArgs`,
whose `entries` tuple can be handed directly from the leader's log to
several followers at once; mutability here would be a subtle way to let
one follower's processing corrupt another's view of the same RPC.
"""

from __future__ import annotations

from dataclasses import dataclass

from .log import LogEntry
from .types import LogIndex, NodeId, Term


@dataclass(frozen=True, slots=True)
class RequestVoteArgs:
    term: Term
    candidate_id: NodeId
    last_log_index: LogIndex
    last_log_term: Term


@dataclass(frozen=True, slots=True)
class RequestVoteReply:
    term: Term
    vote_granted: bool
    voter_id: NodeId


@dataclass(frozen=True, slots=True)
class AppendEntriesArgs:
    term: Term
    leader_id: NodeId
    prev_log_index: LogIndex
    prev_log_term: Term
    entries: tuple[LogEntry, ...]
    leader_commit: LogIndex


@dataclass(frozen=True, slots=True)
class AppendEntriesReply:
    term: Term
    success: bool
    follower_id: NodeId
    match_index: LogIndex


RpcPayload = RequestVoteArgs | RequestVoteReply | AppendEntriesArgs | AppendEntriesReply


@dataclass(frozen=True, slots=True)
class Message:
    """An RPC payload addressed to a specific node. `RaftNode` never sends
    or receives these directly -- `tick()`, `step()`, and `propose()` all
    return `list[Message]` for a transport to actually deliver."""

    src: NodeId
    dst: NodeId
    payload: RpcPayload
