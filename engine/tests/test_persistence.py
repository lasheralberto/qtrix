"""E8·02: the conversation and the report of a notebook come back as they were after a restart."""

from __future__ import annotations

import json
import time

from fastapi.testclient import TestClient
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from qtrix_engine import api
from qtrix_engine.api import create_app
from qtrix_engine.config import Settings
from qtrix_engine.panel import Block

from .conftest import make_wages
from .test_api import client, events


def start(tmp_path, **kwargs) -> TestClient:
    return TestClient(create_app(Settings(home=tmp_path, token=None, local_only=False), with_memory=False, **kwargs))


def seeded(tmp_path):
    """A notebook with a dataset, a chat turn that fitted a model, a note and a Python cell."""
    make_wages().to_csv(tmp_path / "wages.csv", index=False)
    c = client(tmp_path)
    nid = c.post("/notebooks", json={"title": "Salarios"}).json()["id"]
    base = f"/notebooks/{nid}"
    imported = c.post(f"{base}/datasets/import", json={"path": str(tmp_path / "wages.csv"), "name": "wages"})
    assert imported.status_code == 200
    assert events(c.post(f"{base}/chat", json={"message": "return to education?"}))[-1]["type"] == "done"
    assert c.post(f"{base}/panel/notes", json={"markdown": "Hallazgo: **8%**", "title": "Nota"}).status_code == 200
    assert c.post(f"{base}/panel/cells", json={"code": "1 + 1"}).status_code == 200
    return c, nid, base


def test_chat_and_panel_survive_a_restart(tmp_path):
    c, nid, base = seeded(tmp_path)
    history, panel = c.get(f"{base}/chat/history").json(), c.get(f"{base}/panel").json()
    assert history and len(panel["blocks"]) >= 3
    c.app.state.engine.flush()

    again = client(tmp_path)
    assert again.get(f"/notebooks/{nid}/chat/history").json() == history
    restored = again.get(f"/notebooks/{nid}/panel").json()
    assert restored["version"] == panel["version"]
    assert [b["id"] for b in restored["blocks"]] == [b["id"] for b in panel["blocks"]]
    assert {b["kind"] for b in restored["blocks"]} >= {"regression_table", "markdown", "python_cell"}


def test_undo_history_survives_too(tmp_path):
    c, nid, base = seeded(tmp_path)
    before = len(c.get(f"{base}/panel").json()["blocks"])
    c.app.state.engine.flush()
    again = client(tmp_path)
    assert again.post(f"/notebooks/{nid}/panel/undo").json()["changed"] is True  # undoes the Python cell added last
    assert len(again.get(f"/notebooks/{nid}/panel").json()["blocks"]) == before - 1


def test_the_agent_remembers_the_conversation_after_a_restart(tmp_path):
    c, nid, base = seeded(tmp_path)
    c.app.state.engine.flush()
    seen: list[int] = []

    def respond(messages, info: AgentInfo):  # the planner (a plain run): it sees a summary, not the model history
        return ModelResponse(parts=[TextPart(json.dumps([str(m) for m in messages])[:2000])])

    async def stream(messages, info: AgentInfo):
        seen.append(len(messages))
        yield "ok"

    again = start(tmp_path, model_override=FunctionModel(respond, stream_function=stream))
    assert events(again.post(f"/notebooks/{nid}/chat", json={"message": "and with experience?"}))[-1]["type"] == "done"
    assert seen and seen[0] > 1  # the model got the earlier turns, not just the new question


def test_reset_is_saved(tmp_path):
    c, nid, base = seeded(tmp_path)
    c.app.state.engine.flush()
    assert c.post(f"{base}/chat/reset").json() == {"ok": True}
    c.app.state.engine.flush()
    again = client(tmp_path)
    assert again.get(f"/notebooks/{nid}/chat/history").json() == []
    assert len(again.get(f"/notebooks/{nid}/panel").json()["blocks"]) >= 3  # the report is not the chat


def test_a_cell_that_was_running_is_not_running_after_a_restart(tmp_path):
    c, nid, base = seeded(tmp_path)
    ctx = c.app.state.engine.open(nid)
    ctx.panel.add(Block(kind="python_cell", title="Python", data={"running": True}, code="sleep", source="user"))
    c.app.state.engine.flush()
    again = client(tmp_path)
    cells = [b for b in again.get(f"/notebooks/{nid}/panel").json()["blocks"] if b["kind"] == "python_cell"]
    assert cells and all(not b["data"].get("running") for b in cells)


def test_changes_are_saved_by_themselves_without_a_flush(tmp_path):
    c, nid, base = seeded(tmp_path)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:  # the autosave writes a moment after the last change
        if len(client(tmp_path).get(f"/notebooks/{nid}/panel").json()["blocks"]) >= 3:
            break
        time.sleep(0.2)
    assert len(client(tmp_path).get(f"/notebooks/{nid}/panel").json()["blocks"]) >= 3


def test_duplicate_carries_the_conversation_and_the_report(tmp_path):
    c, nid, base = seeded(tmp_path)
    copy = c.post(f"/notebooks/{nid}/duplicate", json={"title": "Copia"}).json()["id"]
    assert c.get(f"/notebooks/{copy}/chat/history").json() == c.get(f"{base}/chat/history").json()
    assert ([b["kind"] for b in c.get(f"/notebooks/{copy}/panel").json()["blocks"]]
            == [b["kind"] for b in c.get(f"{base}/panel").json()["blocks"]])
    c.post(f"/notebooks/{copy}/chat/reset")
    assert c.get(f"{base}/chat/history").json()  # the original keeps its conversation


def test_least_recently_used_notebooks_are_closed_and_come_back_intact(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "MAX_OPEN", 2)
    c, first, base = seeded(tmp_path)
    history = c.get(f"{base}/chat/history").json()
    engine = c.app.state.engine
    for title in ("dos", "tres", "cuatro"):
        c.post("/notebooks", json={"title": title})
    assert len(engine._contexts) <= 2 and first not in engine._contexts  # closed, with its state saved
    assert c.get(f"/notebooks/{first}/chat/history").json() == history  # reopened on demand
    assert any(b["kind"] == "markdown" for b in c.get(f"/notebooks/{first}/panel").json()["blocks"])
    listed = {n["id"]: n for n in c.get("/notebooks").json()}  # listing does not reopen the closed ones
    assert listed[first]["n_datasets"] == 1
