"""
raftkv.transport.codec
=========================

JSON encode/decode for everything that crosses a real socket: Raft RPCs
(`raftkv.raft.rpc`), the log entries embedded inside AppendEntries, and
KV client requests/responses.

Every serializable shape carries an explicit `"type"` tag rather than
being decoded by guessing from its fields -- deliberately boring and
explicit, because getting this wrong would silently hand a node the
wrong RPC and corrupt cluster state in a way none of Raft's safety
properties are designed to detect (they all assume a decoded message is
actually what it claims to be).
"""

from __future__ import annotations

from typing import Any

from ..kv.client import ClientRequest, ClientResponse
from ..kv.store import Command, CompareAndSwapCommand, DeleteCommand, GetCommand, Op, SetCommand
from ..raft.log import LogEntry
from ..raft.rpc import (
    AppendEntriesArgs,
    AppendEntriesReply,
    Message,
    RequestVoteArgs,
    RequestVoteReply,
)

# -- KV command ops -------------------------------------------------------


def encode_op(op: Op) -> dict[str, Any]:
    if isinstance(op, GetCommand):
        return {"type": "Get", "key": op.key}
    if isinstance(op, SetCommand):
        return {"type": "Set", "key": op.key, "value": op.value}
    if isinstance(op, DeleteCommand):
        return {"type": "Delete", "key": op.key}
    if isinstance(op, CompareAndSwapCommand):
        return {"type": "Cas", "key": op.key, "expected": op.expected, "new_value": op.new_value}
    raise TypeError(f"cannot encode op: {op!r}")  # pragma: no cover


def decode_op(payload: dict[str, Any]) -> Op:
    kind = payload["type"]
    if kind == "Get":
        return GetCommand(payload["key"])
    if kind == "Set":
        return SetCommand(payload["key"], payload["value"])
    if kind == "Delete":
        return DeleteCommand(payload["key"])
    if kind == "Cas":
        return CompareAndSwapCommand(payload["key"], payload["expected"], payload["new_value"])
    raise ValueError(f"unknown op type: {kind!r}")


def encode_command(command: Command) -> dict[str, Any]:
    return {
        "client_id": command.client_id,
        "request_id": command.request_id,
        "op": encode_op(command.op),
    }


def decode_command(payload: dict[str, Any]) -> Command:
    return Command(payload["client_id"], payload["request_id"], decode_op(payload["op"]))


# -- log entries -----------------------------------------------------------


def encode_entry(entry: LogEntry) -> dict[str, Any]:
    return {"term": entry.term, "index": entry.index, "command": encode_command(entry.command)}


def decode_entry(payload: dict[str, Any]) -> LogEntry:
    return LogEntry(
        term=payload["term"], index=payload["index"], command=decode_command(payload["command"])
    )


# -- Raft RPC messages ---------------------------------------------------


def encode_message(message: Message) -> dict[str, Any]:
    payload = message.payload
    if isinstance(payload, RequestVoteArgs):
        body: dict[str, Any] = {
            "type": "RequestVoteArgs",
            "term": payload.term,
            "candidate_id": payload.candidate_id,
            "last_log_index": payload.last_log_index,
            "last_log_term": payload.last_log_term,
        }
    elif isinstance(payload, RequestVoteReply):
        body = {
            "type": "RequestVoteReply",
            "term": payload.term,
            "vote_granted": payload.vote_granted,
            "voter_id": payload.voter_id,
        }
    elif isinstance(payload, AppendEntriesArgs):
        body = {
            "type": "AppendEntriesArgs",
            "term": payload.term,
            "leader_id": payload.leader_id,
            "prev_log_index": payload.prev_log_index,
            "prev_log_term": payload.prev_log_term,
            "entries": [encode_entry(e) for e in payload.entries],
            "leader_commit": payload.leader_commit,
        }
    elif isinstance(payload, AppendEntriesReply):
        body = {
            "type": "AppendEntriesReply",
            "term": payload.term,
            "success": payload.success,
            "follower_id": payload.follower_id,
            "match_index": payload.match_index,
        }
    else:
        raise TypeError(f"cannot encode RPC payload: {payload!r}")  # pragma: no cover
    return {"src": message.src, "dst": message.dst, "payload": body}


def decode_message(data: dict[str, Any]) -> Message:
    body = data["payload"]
    kind = body["type"]
    if kind == "RequestVoteArgs":
        payload: Any = RequestVoteArgs(
            body["term"], body["candidate_id"], body["last_log_index"], body["last_log_term"]
        )
    elif kind == "RequestVoteReply":
        payload = RequestVoteReply(body["term"], body["vote_granted"], body["voter_id"])
    elif kind == "AppendEntriesArgs":
        entries = tuple(decode_entry(e) for e in body["entries"])
        payload = AppendEntriesArgs(
            body["term"],
            body["leader_id"],
            body["prev_log_index"],
            body["prev_log_term"],
            entries,
            body["leader_commit"],
        )
    elif kind == "AppendEntriesReply":
        payload = AppendEntriesReply(
            body["term"], body["success"], body["follower_id"], body["match_index"]
        )
    else:
        raise ValueError(f"unknown RPC payload type: {kind!r}")
    return Message(data["src"], data["dst"], payload)


# -- KV client protocol --------------------------------------------------


def encode_client_request(request: ClientRequest) -> dict[str, Any]:
    return {
        "client_id": request.client_id,
        "request_id": request.request_id,
        "op": encode_op(request.op),
    }


def decode_client_request(data: dict[str, Any]) -> ClientRequest:
    return ClientRequest(data["client_id"], data["request_id"], decode_op(data["op"]))


def encode_client_response(response: ClientResponse) -> dict[str, Any]:
    return {
        "ok": response.ok,
        "result": response.result,
        "leader_hint": response.leader_hint,
        "error": response.error,
    }


def decode_client_response(data: dict[str, Any]) -> ClientResponse:
    return ClientResponse(
        ok=data["ok"],
        result=data.get("result"),
        leader_hint=data.get("leader_hint"),
        error=data.get("error"),
    )
