"""The journal: an append-only record of every call that went through the dispatcher (ADR 0002 §6, E7·06).

``_qtrix_journal`` is the table of the ADR. Next to it, ``_qtrix_journal_before`` keeps the before-images that
undo needs (object rows with their versions, page state); they are an implementation detail of undo and replay.
Each entry carries the canonical Python of the call (``print_call``), so the mutating entries, printed in order,
are a runnable script of the session.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .workspace import Workspace

log = logging.getLogger(__name__)

JOURNAL_DDL = """
CREATE TABLE IF NOT EXISTS _qtrix_journal (
    seq BIGINT PRIMARY KEY,
    ts TIMESTAMP NOT NULL,
    actor VARCHAR NOT NULL,
    channel VARCHAR NOT NULL,
    kind VARCHAR NOT NULL,
    code VARCHAR,
    tool VARCHAR,
    args JSON,
    mutating BOOLEAN NOT NULL,
    affected JSON,
    ok BOOLEAN NOT NULL,
    error VARCHAR,
    duration_ms INTEGER,
    turn_id VARCHAR,
    cell_id VARCHAR,
    edited_from BIGINT,
    proposal_id VARCHAR
)
"""
BEFORE_DDL = """
CREATE TABLE IF NOT EXISTS _qtrix_journal_before (
    seq BIGINT NOT NULL,
    kind VARCHAR NOT NULL,
    key VARCHAR NOT NULL,
    payload JSON NOT NULL,
    PRIMARY KEY (seq, kind, key)
)
"""
COLS = ("seq, ts, actor, channel, kind, code, tool, args, mutating, affected, ok, error, duration_ms, turn_id, "
        "cell_id, edited_from, proposal_id")
UNDO_TOOL = "journal_undo"
KERNEL_CHANNEL = "kernel"  # calls made from inside a Python cell (E7·07b)


class JournalEntryNotFound(KeyError):
    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


# ---- canonical code -------------------------------------------------------------------------------------------
def print_call(tool: str, args: dict[str, Any]) -> str:
    """Canonical ``qx`` Python for a call, e.g. ``eq01 = qx.equation("eq01", page="wages", method="ls", ...)``.

    The printer lives in ``qx.code`` (E7·07); it is imported lazily because the ``qx`` package imports the dispatcher.
    """
    from .qx.code import print_call as qx_print_call

    return qx_print_call(tool, args)


def _json(value: Any) -> str:
    return json.dumps(value, default=str)


class Journal:
    """Append-only journal on the workfile's DuckDB (shares the workspace connection and lock)."""

    def __init__(self, workspace: Workspace) -> None:
        self._ws = workspace
        self._lock = workspace._lock
        self.mutation_lock = threading.RLock()  # mutating calls run one at a time, so before/after states are exact
        self._seq_lock = threading.Lock()
        self._next: int | None = None
        self._listeners: list[Callable[[dict[str, Any]], None]] = []

    # ---- events --------------------------------------------------------------------------------------------
    def subscribe(self, listener: Callable[[dict[str, Any]], None]) -> None:
        self._listeners.append(listener)

    def _emit(self, entry: dict[str, Any]) -> None:
        event = {"event": "journal", **{k: entry[k] for k in ("seq", "tool", "actor", "channel", "mutating", "ok")}}
        for listener in list(self._listeners):
            try:
                listener(event)
            except Exception:  # noqa: BLE001 - a broken listener must never break a call
                log.exception("Journal listener failed")

    # ---- writes ----------------------------------------------------------------------------------------------
    def reserve(self) -> int:
        with self._seq_lock:
            if self._next is None:
                with self._lock:
                    self._next = self._ws._con.execute("SELECT COALESCE(MAX(seq), 0) FROM _qtrix_journal").fetchone()[0]
            self._next += 1
            return self._next

    def append(self, *, seq: int, actor: str, channel: str, tool: str, args: dict[str, Any], mutating: bool,
               affected: list[dict[str, Any]], ok: bool, error: str | None, duration_ms: int, kind: str = "api",
               turn_id: str | None = None, cell_id: str | None = None, edited_from: int | None = None,
               proposal_id: str | None = None, code: str | None = None) -> dict[str, Any]:
        """``code`` defaults to the canonical qx code of the call; a Python cell passes its source."""
        ts = datetime.now(UTC).replace(tzinfo=None)
        if code is None:
            code = print_call(tool, args)
        with self._lock:
            self._ws._con.execute(
                f"INSERT INTO _qtrix_journal ({COLS}) VALUES ({', '.join('?' * 17)})",
                [seq, ts, actor, channel, kind, code, tool, _json(args), mutating, _json(affected), ok, error,
                 duration_ms, turn_id, cell_id, edited_from, proposal_id])
        entry = self.get(seq)
        self._emit(entry)
        return entry

    def save_before(self, seq: int, kind: str, key: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self._ws._con.execute("INSERT OR REPLACE INTO _qtrix_journal_before VALUES (?, ?, ?, ?)",
                                  [seq, kind, key, _json(payload)])

    def discard_before(self, seq: int) -> None:
        """Forget the before-images of a call that failed (their undo tables go too)."""
        with self._lock:
            for (key,) in self._ws._con.execute("SELECT key FROM _qtrix_journal_before WHERE seq = ? AND kind = 'page'",
                                                [seq]).fetchall():
                self._ws._con.execute(f'DROP TABLE IF EXISTS "undo_{seq}_{key}"')
            self._ws._con.execute("DELETE FROM _qtrix_journal_before WHERE seq = ?", [seq])

    def load_before(self, seq: int, kind: str, key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._ws._con.execute("SELECT payload FROM _qtrix_journal_before WHERE seq = ? AND kind = ? "
                                        "AND key = ?", [seq, kind, key]).fetchone()
        return json.loads(row[0]) if row else None

    # ---- reads -----------------------------------------------------------------------------------------------
    @staticmethod
    def _entry(row: tuple) -> dict[str, Any]:
        return {"seq": row[0], "ts": row[1].isoformat(), "actor": row[2], "channel": row[3], "kind": row[4],
                "code": row[5], "tool": row[6], "args": json.loads(row[7]) if row[7] else {}, "mutating": row[8],
                "affected": json.loads(row[9]) if row[9] else [], "ok": row[10], "error": row[11],
                "duration_ms": row[12], "turn_id": row[13], "cell_id": row[14], "edited_from": row[15],
                "proposal_id": row[16]}

    def get(self, seq: int) -> dict[str, Any]:
        with self._lock:
            row = self._ws._con.execute(f"SELECT {COLS} FROM _qtrix_journal WHERE seq = ?", [seq]).fetchone()
        if row is None:
            raise JournalEntryNotFound(f"No journal entry with seq {seq}")
        return self._entry(row)

    def list(self, *, actor: str | None = None, mutating: bool | None = None, since: int = 0, ok: bool | None = None,
             tool: str | None = None, limit: int | None = None, turn_id: str | None = None) -> list[dict[str, Any]]:
        """Entries in order. ``actor='agent'`` matches every ``agent:<name>``; ``since`` is an exclusive seq."""
        where, params = ["seq > ?"], [since]
        if actor:
            where.append("(actor = ? OR actor LIKE ?)")
            params += [actor, actor + ":%"]
        for column, value in (("mutating", mutating), ("ok", ok), ("tool", tool), ("turn_id", turn_id)):
            if value is not None:
                where.append(f"{column} = ?")
                params.append(value)
        sql = f"SELECT {COLS} FROM _qtrix_journal WHERE {' AND '.join(where)} ORDER BY seq"
        if limit:
            sql += f" LIMIT {int(limit)}"
        with self._lock:
            rows = self._ws._con.execute(sql, params).fetchall()
        undone = self.undone()
        return [{**e, "undone_by": undone.get(e["seq"])} for e in map(self._entry, rows)]

    def last_seq(self) -> int:
        with self._lock:
            return self._ws._con.execute("SELECT COALESCE(MAX(seq), 0) FROM _qtrix_journal").fetchone()[0]

    def undone(self) -> dict[int, int]:
        """Entries that were undone, by the seq of the undo entry that did it."""
        with self._lock:
            rows = self._ws._con.execute("SELECT seq, args FROM _qtrix_journal WHERE tool = ? AND ok",
                                         [UNDO_TOOL]).fetchall()
        return {int(json.loads(a)["seq"]): s for s, a in rows}

    def script(self) -> str:
        """The mutating journal, in order, as a Python script. Undone calls are kept as comments.

        A Python cell appears as its source; the calls it made (channel ``kernel``) are left out, since running the
        cell makes them again. A cell that changed the workfile and then failed is kept, flagged.
        """
        undone = self.undone()
        lines = ["# qtrix session script: the mutating journal in order (E7·06)",
                 "import qtrix as qx", ""]
        for e in self.list(mutating=True):
            if e["tool"] == UNDO_TOOL or e["channel"] == KERNEL_CHANNEL or not (e["ok"] or e["kind"] == "python"):
                continue
            origin = f"{e['actor']} via {e['channel']}" + (f" · edited from [{e['edited_from']}]"
                                                           if e["edited_from"] else "")
            lines.append(f"# [{e['seq']}] {e['ts']} · {origin}" + ("" if e["ok"] else f" · FAILED: {e['error']}"))
            if e["seq"] in undone:
                lines.append(f"# undone by [{undone[e['seq']]}]:")
                lines.extend(f"#   {ln}" for ln in e["code"].splitlines())
            else:
                lines.append(e["code"])
        return "\n".join(lines) + "\n"
