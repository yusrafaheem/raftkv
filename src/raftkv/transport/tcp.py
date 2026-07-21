"""
raftkv.transport.tcp
=======================

A real, runnable transport: each `RaftServer` is one OS process running
one `RaftNode`, talking to its peers over plain TCP with a
newline-delimited-JSON wire format (`raftkv.transport.codec`). This is
the counterpart to `raftkv.transport.simulated.SimulatedNetwork` --
same `RaftNode` core, same `tick()`/`step()`/`propose()` contract, real
sockets and a real OS clock instead of a deterministic discrete-event
loop.

Concurrency model: everything here runs on a single asyncio event loop
per process. `RaftNode` mutation only ever happens inside `await`-free
sections of `_tick_loop`, `_handle_peer_conn`, and
`_handle_client_request`, so a plain `asyncio.Lock` is enough to keep
those sections from interleaving -- there's no possibility of two OS
threads touching `self.node` at once, only of two coroutines
interleaving at an `await` point, which the lock rules out.

What this deliberately does not do (see README's scope notes for the
fuller list): persist any Raft state to disk, so a real process crash
loses `current_term` / `voted_for` / the log -- correct as a Raft
peer only survives *simulated* crashes (see `cluster.py`), not real
ones. A production node would fsync all of Figure 2's persistent state
before replying to any RPC.
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket

from ..kv.client import ClientRequest, ClientResponse, SendFn
from ..kv.store import Command, KVStateMachine
from ..raft.node import RaftNode
from ..raft.rpc import Message
from ..raft.types import NodeId, Role
from .codec import (
    decode_client_request,
    decode_client_response,
    decode_message,
    encode_client_request,
    encode_client_response,
    encode_message,
)

logger = logging.getLogger("raftkv.transport.tcp")

DEFAULT_TICK_INTERVAL_SECONDS = 0.05  # 50ms logical ticks
DEFAULT_COMMIT_WAIT_SECONDS = 5.0
DEFAULT_CONNECT_TIMEOUT_SECONDS = 0.5


class RaftServer:
    """Runs one `RaftNode` as an asyncio TCP service: a peer-RPC listener,
    a KV client-request listener, and a background tick loop."""

    def __init__(
        self,
        node_id: NodeId,
        peers: dict[NodeId, tuple[str, int]],
        *,
        peer_host: str,
        peer_port: int,
        client_host: str = "127.0.0.1",
        client_port: int,
        tick_interval: float = DEFAULT_TICK_INTERVAL_SECONDS,
        commit_wait_seconds: float = DEFAULT_COMMIT_WAIT_SECONDS,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT_SECONDS,
        node: RaftNode | None = None,
    ) -> None:
        self.node_id = node_id
        self.peers = dict(peers)  # node_id -> (host, port) of that peer's *peer* port
        self.peer_host = peer_host
        self.peer_port = peer_port
        self.client_host = client_host
        self.client_port = client_port
        self.tick_interval = tick_interval
        self.commit_wait_seconds = commit_wait_seconds
        self.connect_timeout = connect_timeout

        self.node = node or RaftNode(node_id, list(peers.keys()))
        self.state_machine = KVStateMachine()

        self._peer_server: asyncio.Server | None = None
        self._client_server: asyncio.Server | None = None
        self._tick_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        # (log_index, term) -> Future resolved with the applied command's
        # result once that exact entry commits. Keying on term as well as
        # index means a request whose entry gets overwritten by a new
        # leader (different term at the same index) is never mistakenly
        # resolved with someone else's result -- it just times out, which
        # is correct: the client can't tell that happened either, and its
        # retry will land on whichever entry actually won.
        self._pending: dict[tuple[int, int], asyncio.Future] = {}

    async def start(self) -> None:
        self._peer_server = await asyncio.start_server(
            self._handle_peer_conn, self.peer_host, self.peer_port
        )
        self._client_server = await asyncio.start_server(
            self._handle_client_conn, self.client_host, self.client_port
        )
        self._tick_task = asyncio.create_task(self._tick_loop())

    async def stop(self) -> None:
        if self._tick_task is not None:
            self._tick_task.cancel()
            try:
                await self._tick_task
            except asyncio.CancelledError:
                pass
        for server in (self._peer_server, self._client_server):
            if server is not None:
                server.close()
                await server.wait_closed()
        for future in self._pending.values():
            if not future.done():
                future.cancel()
        self._pending.clear()

    # -- background tick loop -------------------------------------------------

    async def _tick_loop(self) -> None:
        while True:
            await asyncio.sleep(self.tick_interval)
            async with self._lock:
                messages = self.node.tick()
                self._apply_committed()
            await self._deliver_all(messages)

    def _apply_committed(self) -> None:
        for entry in self.node.take_committed_entries():
            result = self.state_machine.apply(entry.command)
            future = self._pending.pop((entry.index, entry.term), None)
            if future is not None and not future.done():
                future.set_result(result)

    # -- outbound RPC delivery -------------------------------------------------

    async def _deliver_all(self, messages: list[Message]) -> None:
        for message in messages:
            asyncio.create_task(self._deliver_one(message))

    async def _deliver_one(self, message: Message) -> None:
        addr = self.peers.get(message.dst)
        if addr is None:
            return
        host, port = addr
        writer = None
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=self.connect_timeout
            )
            writer.write((json.dumps(encode_message(message)) + "\n").encode())
            await writer.drain()
        except (OSError, asyncio.TimeoutError):
            # Peer unreachable right now -- Raft is designed to tolerate
            # this. Heartbeats retry on the next tick, and a follower
            # that missed entries will get them re-sent once its
            # AppendEntriesReply (or the absence of one) is next handled.
            pass
        finally:
            if writer is not None:
                writer.close()

    # -- inbound peer RPCs -----------------------------------------------------

    async def _handle_peer_conn(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            line = await reader.readline()
            if not line:
                return
            message = decode_message(json.loads(line.decode()))
            async with self._lock:
                outbound = self.node.step(message)
                self._apply_committed()
            await self._deliver_all(outbound)
        except Exception:
            logger.exception("error handling peer connection")
        finally:
            writer.close()

    # -- inbound KV client requests --------------------------------------------

    async def _handle_client_conn(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            line = await reader.readline()
            if not line:
                return
            request = decode_client_request(json.loads(line.decode()))
            response = await self._handle_client_request(request)
            writer.write((json.dumps(encode_client_response(response)) + "\n").encode())
            await writer.drain()
        except Exception:
            logger.exception("error handling client connection")
        finally:
            writer.close()

    async def _handle_client_request(self, request: ClientRequest) -> ClientResponse:
        async with self._lock:
            if self.node.role is not Role.LEADER:
                return ClientResponse(ok=False, leader_hint=self.node.leader_id, error="not leader")

            command = Command(request.client_id, request.request_id, request.op)
            result = self.node.propose(command)
            if result.index is None:
                return ClientResponse(
                    ok=False, leader_hint=self.node.leader_id, error="propose rejected"
                )

            future: asyncio.Future = asyncio.get_event_loop().create_future()
            self._pending[(result.index, result.term)] = future
            outbound = list(result.messages)

        await self._deliver_all(outbound)

        try:
            value = await asyncio.wait_for(future, timeout=self.commit_wait_seconds)
            return ClientResponse(ok=True, result=value)
        except asyncio.TimeoutError:
            self._pending.pop((result.index, result.term), None)
            return ClientResponse(ok=False, error="commit timed out")


def tcp_sender(
    node_addrs: dict[NodeId, tuple[str, int]], *, timeout: float = 2.0
) -> SendFn:
    """A blocking, socket-based `KVClient` sender that talks to a real
    `RaftServer` cluster's client ports. Deliberately synchronous (plain
    `socket`, not asyncio) since `KVClient` itself is synchronous -- this
    is what `raftkv-cli` uses, and it's exactly the same shape of
    callable `raftkv.cluster.make_simulated_sender` builds for tests, so
    `KVClient`'s retry/leader-discovery logic is identical in both
    worlds."""

    def send(node_id: NodeId, request: ClientRequest) -> ClientResponse | None:
        addr = node_addrs.get(node_id)
        if addr is None:
            return None
        host, port = addr
        try:
            with socket.create_connection((host, port), timeout=timeout) as sock:
                sock.sendall((json.dumps(encode_client_request(request)) + "\n").encode())
                sock.shutdown(socket.SHUT_WR)
                buf = b""
                while b"\n" not in buf:
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    buf += chunk
            if not buf:
                return None
            line = buf.split(b"\n", 1)[0]
            return decode_client_response(json.loads(line.decode()))
        except (OSError, TimeoutError):
            return None

    return send
