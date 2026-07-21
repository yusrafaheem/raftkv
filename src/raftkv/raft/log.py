"""
raftkv.raft.log
==================

The replicated log itself: an ordered sequence of `LogEntry`, each
tagged with the term it was created in (this is what lets a node tell a
stale entry from a current one during the AppendEntries consistency
check) and the 1-based index it occupies.

Indices are 1-based to match the Raft paper's own convention -- index 0
is reserved as "before the start of the log", so `prevLogIndex=0` in an
AppendEntries RPC unambiguously means "start replicating from the very
first entry" without needing a separate sentinel value.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import LogIndex, Term


@dataclass(frozen=True, slots=True)
class LogEntry:
    term: Term
    index: LogIndex
    command: object  # opaque to Raft; interpreted by the state machine once committed


class RaftLog:
    """A 1-indexed append/truncate log.

    Internally backed by a plain Python list where `_entries[i]` holds
    the entry at Raft index `i + 1`. All public methods speak in Raft's
    1-based indices so callers never have to think about the off-by-one
    translation.
    """

    def __init__(self) -> None:
        self._entries: list[LogEntry] = []

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def last_index(self) -> LogIndex:
        return len(self._entries)

    def last_term(self) -> Term:
        return self._entries[-1].term if self._entries else 0

    def term_at(self, index: LogIndex) -> Term:
        """Term of the entry at `index`, or 0 if `index` is 0 or past the
        end of the log. Returning 0 rather than raising lets the
        AppendEntries consistency check (`leader.prevLogTerm ==
        follower.term_at(prevLogIndex)`) stay a single comparison instead
        of a try/except at every call site.
        """
        if index <= 0 or index > len(self._entries):
            return 0
        return self._entries[index - 1].term

    def get(self, index: LogIndex) -> LogEntry | None:
        if index <= 0 or index > len(self._entries):
            return None
        return self._entries[index - 1]

    def entries_from(self, index: LogIndex) -> list[LogEntry]:
        """All entries at or after `index` (used by the leader to build the
        `entries` payload of an AppendEntries RPC)."""
        if index <= 0:
            return list(self._entries)
        return self._entries[index - 1 :]

    def entries_between(self, start: LogIndex, end: LogIndex) -> list[LogEntry]:
        """Entries with index in `[start, end]` inclusive (used to pull out
        newly-committed entries for application to the state machine)."""
        if start <= 0:
            start = 1
        return self._entries[start - 1 : end]

    def append(self, entry: LogEntry) -> None:
        expected = len(self._entries) + 1
        if entry.index != expected:
            raise ValueError(
                f"log append out of order: expected index {expected}, got {entry.index}"
            )
        self._entries.append(entry)

    def truncate_from(self, index: LogIndex) -> None:
        """Delete every entry at or after `index` -- used when a follower
        discovers its log conflicts with the leader's (Raft section 5.3:
        "If an existing entry conflicts with a new one ... delete the
        existing entry and all that follow it")."""
        if index <= 0:
            self._entries.clear()
            return
        del self._entries[index - 1 :]
