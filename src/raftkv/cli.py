"""
raftkv.cli
=============

Command-line entry points that make raftkv something you can actually
run, not just unit-test: `raftkv-node` starts one cluster member as a
real OS process talking real TCP; `raftkv-cli` is a thin client for
poking a running cluster from the shell. See the README's "Running a
real cluster" section for a full three-terminal walkthrough.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from .kv.client import ClientTimeoutError, KVClient
from .transport.tcp import DEFAULT_TICK_INTERVAL_SECONDS, RaftServer, tcp_sender


def _parse_addr_arg(value: str) -> tuple[int, str, int]:
    """Parse one `ID=HOST:PORT` argument."""
    try:
        node_part, addr_part = value.split("=", 1)
        host, port_str = addr_part.rsplit(":", 1)
        return int(node_part), host, int(port_str)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected ID=HOST:PORT, got {value!r}") from None


def main_node(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raftkv-node", description="Run one raftkv cluster member."
    )
    parser.add_argument("--id", type=int, required=True, help="this node's numeric id")
    parser.add_argument(
        "--peer",
        action="append",
        required=True,
        metavar="ID=HOST:PORT",
        help="a cluster member's peer (Raft RPC) address; repeat once per node, "
        "including this node itself",
    )
    parser.add_argument(
        "--client-port", type=int, required=True, help="port to listen for KV client requests on"
    )
    parser.add_argument("--client-host", default="127.0.0.1")
    parser.add_argument("--tick-interval", type=float, default=DEFAULT_TICK_INTERVAL_SECONDS)
    args = parser.parse_args(argv)

    all_peers = {nid: (host, port) for nid, host, port in (_parse_addr_arg(p) for p in args.peer)}
    if args.id not in all_peers:
        parser.error(f"--id {args.id} must also appear in one of the --peer entries")
    self_host, self_port = all_peers.pop(args.id)

    server = RaftServer(
        node_id=args.id,
        peers=all_peers,
        peer_host=self_host,
        peer_port=self_port,
        client_host=args.client_host,
        client_port=args.client_port,
        tick_interval=args.tick_interval,
    )

    async def run() -> None:
        await server.start()
        print(
            f"raftkv node {args.id}: peers={sorted(all_peers)} "
            f"peer-addr={self_host}:{self_port} client-addr={args.client_host}:{args.client_port}",
            file=sys.stderr,
        )
        try:
            await asyncio.Event().wait()  # run forever until cancelled/interrupted
        finally:
            await server.stop()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
    return 0


def main_client(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raftkv-cli", description="Talk to a running raftkv cluster."
    )
    parser.add_argument(
        "--node",
        action="append",
        required=True,
        metavar="ID=HOST:PORT",
        help="a cluster member's client-port address; repeat once per node",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    get_p = sub.add_parser("get", help="read a key")
    get_p.add_argument("key")

    set_p = sub.add_parser("set", help="write a key")
    set_p.add_argument("key")
    set_p.add_argument("value")

    del_p = sub.add_parser("delete", help="delete a key")
    del_p.add_argument("key")

    cas_p = sub.add_parser(
        "cas", help="compare-and-swap: set `key` to new_value only if it currently equals expected"
    )
    cas_p.add_argument("key")
    cas_p.add_argument("expected", help='use the literal "(nil)" to mean "key must not exist"')
    cas_p.add_argument("new_value")

    args = parser.parse_args(argv)
    node_addrs = {nid: (host, port) for nid, host, port in (_parse_addr_arg(n) for n in args.node)}

    send = tcp_sender(node_addrs)
    client = KVClient(list(node_addrs.keys()), send)

    try:
        if args.command == "get":
            value = client.get(args.key)
            print(value if value is not None else "(nil)")
        elif args.command == "set":
            client.set(args.key, args.value)
            print("OK")
        elif args.command == "delete":
            client.delete(args.key)
            print("OK")
        elif args.command == "cas":
            expected = None if args.expected == "(nil)" else args.expected
            ok = client.compare_and_swap(args.key, expected, args.new_value)
            print("OK" if ok else "FAILED (current value did not match `expected`)")
    except ClientTimeoutError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main_node())
