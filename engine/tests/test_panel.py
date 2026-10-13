import asyncio

import pytest

from qtrix_engine.panel import Block, PanelState, PinnedBlockError


def _block(title="t"):
    return Block(kind="markdown", title=title, data={"markdown": "x"})


def test_ops_versioning_and_undo():
    p = PanelState()
    a, b = p.add(_block("a")), p.add(_block("b"))
    p.update(a.id, title="a2")
    p.reorder([b.id, a.id])
    assert [x.title for x in p.blocks()] == ["b", "a2"] and p.version == 4
    assert p.undo() and [x.title for x in p.blocks()] == ["a2", "b"]
    assert p.redo() and [x.title for x in p.blocks()] == ["b", "a2"]


def test_pinned_blocks_are_protected_from_agents():
    p = PanelState()
    a = p.add(_block())
    p.pin(a.id)
    with pytest.raises(PinnedBlockError):
        p.update(a.id, title="x")
    with pytest.raises(PinnedBlockError):
        p.remove(a.id)
    p.remove(a.id, by_agent=False)
    assert p.blocks() == []


def test_clear_keeps_pinned():
    p = PanelState()
    a, _ = p.add(_block("keep")), p.add(_block("drop"))
    p.pin(a.id)
    p.clear()
    assert [x.title for x in p.blocks()] == ["keep"]


async def test_subscribers_receive_ops_from_threads():
    p = PanelState()
    q = p.subscribe()
    await asyncio.to_thread(p.add, _block("from worker"))
    op = await asyncio.wait_for(q.get(), 1)
    assert op.op == "add" and op.block.title == "from worker"
