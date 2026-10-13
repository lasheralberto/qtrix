"""journal_* tools (internal): act on the journal itself. Reachable through the dispatcher, never offered to agents."""

from __future__ import annotations

from typing import Any

from .registry import Qx, tool


@tool("journal", mutating=True, internal=True, qx=Qx("qx", "undo", ("seq",)))
def journal_undo(seq: int) -> dict[str, Any]:
    """Undo one journaled call: restore the object versions and page state it changed.

    Args:
        seq: Sequence number of the journal entry to undo.
    """
    from ..replay import undo

    return undo(seq)
