"""The user's own panel blocks (E7·08d): notes and cells written in the panel, which agents never touch."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from qtrix_engine.agents.lead import workspace_context
from qtrix_engine.api import create_app
from qtrix_engine.config import Settings
from qtrix_engine.panel import Block, PanelState, PinnedBlockError

EQ = 'qx.equation("eq_u", page="wages", method="ls", formula="lwage ~ educ")'


@pytest.fixture
def client(tmp_path, wages_csv):
    c = TestClient(create_app(Settings(home=tmp_path, token=None, local_only=False), with_memory=False))
    c.post("/datasets/import", json={"path": str(wages_csv)})
    return c


def test_agents_cannot_modify_or_remove_user_blocks():
    panel = PanelState()
    note = panel.add(Block(kind="markdown", title="Mi nota", data={"markdown": "hola"}, source="user"))
    with pytest.raises(PinnedBlockError, match="written by the user"):
        panel.update(note.id, data={"markdown": "otra"})
    with pytest.raises(PinnedBlockError, match="written by the user"):
        panel.remove(note.id)
    assert panel.update(note.id, data={"markdown": "otra"}, by_agent=False).data == {"markdown": "otra"}
    panel.remove(note.id, by_agent=False)
    assert panel.blocks() == []


def test_notes_are_added_edited_and_shown_to_the_agent(client):
    note = client.post("/panel/notes", json={"title": "Hipótesis", "markdown": "El retorno es **no lineal**"}).json()
    assert note["kind"] == "markdown" and note["source"] == "user"
    edited = client.put(f"/panel/blocks/{note['id']}/note", json={"title": "Hipótesis", "markdown": "Lineal"}).json()
    assert edited["data"] == {"markdown": "Lineal"}
    context = workspace_context()
    assert f"[{note['id']}] markdown: Hipótesis BY USER" in context and "   Lineal" in context
    # An agent block cannot be edited as a note.
    client.post("/tools/ui_add_note", json={"title": "del agente", "markdown": "x"})
    agent_note = client.get("/panel").json()["blocks"][-1]
    assert client.put(f"/panel/blocks/{agent_note['id']}/note", json={"markdown": "y"}).status_code == 400
    # One undo step per edit.
    client.post("/panel/undo")
    client.post("/panel/undo")
    assert client.get("/panel").json()["blocks"][0]["data"] == {"markdown": "El retorno es **no lineal**"}


def test_cells_run_in_the_panel_before_what_they_produce(client):
    block = client.post("/panel/cells", json={"code": EQ}).json()
    assert block["kind"] == "python_cell" and block["source"] == "user" and block["code"] == EQ
    run = block["data"]["run"]
    assert run["kind"] == "api" and run["ok"] and run["versions"] == [{"name": "eq_u", "version": 1}]
    blocks = client.get("/panel").json()["blocks"]
    assert blocks[0]["id"] == block["id"] and blocks[1]["ref"]["object"] == "eq_u"  # the cell first, then its table
    entry = client.get(f"/journal/{run['calls'][0]['seq']}").json()
    assert (entry["actor"], entry["cell_id"]) == ("user", block["id"])

    # Re-run edited: a new version linked to the previous run.
    edited = EQ.replace("lwage ~ educ", "lwage ~ educ + exper")
    again = client.put(f"/panel/blocks/{block['id']}/cell", json={"code": edited}).json()
    assert again["code"] == edited and again["data"]["run"]["versions"] == [{"name": "eq_u", "version": 2}]
    assert client.get(f"/journal/{again['data']['run']['calls'][0]['seq']}").json()["edited_from"] == entry["seq"]


def test_a_failing_cell_keeps_its_error(client):
    block = client.post("/panel/cells", json={"code": 'qx.equation("eq_x", page="nope", formula="y ~ x")'}).json()
    run = block["data"]["run"]
    assert run["ok"] is False and run["calls"][0]["error"]
