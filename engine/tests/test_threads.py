"""Conversations (threads) of a notebook: each has its chat and its report; the data are the notebook's."""

from __future__ import annotations

import duckdb

from qtrix_engine.agents.lead import workspace_context
from qtrix_engine.context import _active, activate

from .test_api import client, events
from .test_persistence import seeded


def threads(c, base):
    return c.get(f"{base}/threads").json()


def test_a_notebook_starts_with_one_conversation_named_after_its_first_question(tmp_path):
    c, nid, base = seeded(tmp_path)
    (only,) = threads(c, base)
    assert only["current"] and only["n_turns"] == 1 and only["title"] == "return to education?"


def test_a_new_conversation_is_empty_but_works_on_the_same_data(tmp_path):
    c, nid, base = seeded(tmp_path)
    first = threads(c, base)[0]["id"]
    new = c.post(f"{base}/threads", json={}).json()
    assert new["current"] and new["n_turns"] == 0 and new["id"] != first
    tb = f"{base}/threads/{new['id']}"
    assert c.get(f"{tb}/chat/history").json() == [] and c.get(f"{tb}/panel").json()["blocks"] == []
    assert [d["name"] for d in c.get(f"{tb}/datasets").json()] == ["wages"]  # the notebook's datasets

    assert events(c.post(f"{tb}/chat", json={"message": "return to education again?"}))[-1]["type"] == "done"
    assert c.get(f"{tb}/chat/history").json()
    assert any(b["kind"] == "regression_table" for b in c.get(f"{tb}/panel").json()["blocks"])
    first_turns = [m for m in c.get(f"{base}/threads/{first}/chat/history").json() if m["role"] == "user"]
    assert len(first_turns) == 1  # the first conversation did not grow
    assert {t["id"]: t["title"] for t in threads(c, base)}[new["id"]] == "return to education again?"


def test_unprefixed_routes_follow_the_conversation_open_in_the_ui(tmp_path):
    c, nid, base = seeded(tmp_path)
    first = threads(c, base)[0]["id"]
    c.post(f"{base}/threads", json={"title": "Robustez"})
    assert c.get(f"{base}/chat/history").json() == []  # the new conversation is the one in use
    c.post(f"{base}/threads/{first}/open")
    assert c.get(f"{base}/chat/history").json() != []


def test_conversations_survive_a_restart(tmp_path):
    c, nid, base = seeded(tmp_path)
    new = c.post(f"{base}/threads", json={"title": "Robustez"}).json()["id"]
    assert c.post(f"{base}/threads/{new}/panel/notes", json={"markdown": "Otra nota"}).status_code == 200
    before = threads(c, base)
    c.app.state.engine.flush()

    again = client(tmp_path)
    after = again.get(f"/notebooks/{nid}/threads").json()
    assert ([(t["id"], t["title"], t["n_blocks"]) for t in after]
            == [(t["id"], t["title"], t["n_blocks"]) for t in before])
    assert again.get(f"/notebooks/{nid}/threads/{before[0]['id']}/chat/history").json()


def test_rename_and_delete(tmp_path):
    c, nid, base = seeded(tmp_path)
    first = threads(c, base)[0]["id"]
    assert c.delete(f"{base}/threads/{first}").status_code == 409  # a notebook keeps at least one
    new = c.post(f"{base}/threads", json={}).json()["id"]
    assert c.patch(f"{base}/threads/{new}", json={"title": "Placebo"}).json()["title"] == "Placebo"
    assert c.delete(f"{base}/threads/{new}").json() == {"ok": True}
    assert [t["id"] for t in threads(c, base)] == [first]
    assert threads(c, base)[0]["current"]
    assert c.get(f"{base}/threads/{new}/chat/history").status_code == 404
    c.app.state.engine.flush()
    assert [t["id"] for t in client(tmp_path).get(f"/notebooks/{nid}/threads").json()] == [first]


def test_object_changes_reach_the_report_of_every_conversation(tmp_path):
    c, nid, base = seeded(tmp_path)
    ctx = c.app.state.engine.open(nid)
    first = next(iter(ctx.threads.values()))
    c.post(f"{base}/threads", json={})  # the other conversation is now the one in use
    name = next(b.result_id for b in first.panel.blocks() if b.result_id)
    ctx.objects.rename(name, "renamed_eq")
    assert any(b.result_id == "renamed_eq" for b in first.panel.blocks())


def test_a_notebook_saved_before_threads_becomes_its_first_conversation(tmp_path):
    c, nid, base = seeded(tmp_path)
    history = c.get(f"{base}/chat/history").json()
    blocks = [b["id"] for b in c.get(f"{base}/panel").json()["blocks"]]
    engine = c.app.state.engine
    db = engine.notebooks.get(nid).db_path
    engine._close(engine._contexts[nid])  # saves it
    con = duckdb.connect(str(db))  # the layout of earlier versions: one panel row, no thread list
    con.execute("INSERT OR REPLACE INTO _qtrix_panel SELECT 1, version, blocks, history, redo, updated_at "
                "FROM _qtrix_panels WHERE thread_id = 'main'")
    con.execute("DROP TABLE _qtrix_panels")
    con.execute("DROP TABLE _qtrix_threads")
    con.close()

    again = client(tmp_path)
    (only,) = again.get(f"/notebooks/{nid}/threads").json()
    assert only["id"] == "main" and only["title"] == "return to education?"
    assert again.get(f"/notebooks/{nid}/chat/history").json() == history
    assert [b["id"] for b in again.get(f"/notebooks/{nid}/panel").json()["blocks"]] == blocks


def test_a_new_conversation_is_told_about_the_objects_the_others_built(tmp_path):
    c, nid, base = seeded(tmp_path)
    ctx = c.app.state.engine.open(nid)
    c.post(f"{base}/threads", json={})
    token = activate(ctx)
    try:
        text = workspace_context()
    finally:
        _active.reset(token)
    assert ctx.objects.list()[0].name in text and "## Analysis panel of this conversation (in order)\n(empty)" in text
