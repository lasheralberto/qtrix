"""Notebooks (E8·01): one isolated container per working session — datasets, objects, journal, panel, chat."""

from __future__ import annotations

import socket
import threading
import time

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient

from qtrix_engine.api import create_app
from qtrix_engine.config import Settings
from qtrix_engine.notebooks import NotebookManager, NotebookNotFound
from qtrix_engine.workspace import Workspace

from .conftest import make_macro, make_wages
from .test_api import client, events


def prefixed(c: TestClient, nid: str, path: str) -> str:
    return f"/notebooks/{nid}{path}"


@pytest.fixture
def two(tmp_path):
    """An app with two notebooks: ``a`` holds a wage panel and ``b`` a macro series, both named ``d``."""
    make_wages().to_csv(tmp_path / "wages.csv", index=False)
    make_macro().to_csv(tmp_path / "macro.csv", index=False)
    c = client(tmp_path)
    initial = c.get("/notebooks").json()[0]["id"]  # the empty "Mi primer cuaderno" of a fresh folder
    a = c.post("/notebooks", json={"title": "Salarios"}).json()["id"]
    b = c.post("/notebooks", json={"title": "Macro"}).json()["id"]
    c.delete(f"/notebooks/{initial}")
    for nid, csv in ((a, "wages.csv"), (b, "macro.csv")):
        r = c.post(prefixed(c, nid, "/datasets/import"), json={"path": str(tmp_path / csv), "name": "d"})
        assert r.status_code == 200, r.text
    return c, a, b


def test_manager_lifecycle(tmp_path):
    mgr = NotebookManager(tmp_path / "notebooks")
    assert mgr.list() == []
    first = mgr.create("  Uno  ", "desc")
    assert first.title == "Uno" and first.db_path.name == "notebook.duckdb" and first.uploads_dir.is_dir()
    second = mgr.create()
    assert second.title == "Cuaderno sin título"
    mgr.touch(first.id)
    assert [n.id for n in mgr.list()][0] == first.id  # last opened first
    assert mgr.update(first.id, title="Dos").title == "Dos"
    assert mgr.update(first.id, title="   ").title == "Dos"  # a blank title is ignored
    trash = mgr.remove(second.id)
    assert trash.parent.name == ".trash" and trash.is_dir()
    assert [n.id for n in mgr.list()] == [first.id]
    for bad in ("nb-missing", "../escape", "", "A B"):
        with pytest.raises(NotebookNotFound):
            mgr.get(bad)


def test_legacy_workspace_becomes_the_first_notebook(tmp_path):
    settings = Settings(home=tmp_path, token=None, local_only=False)
    legacy = Workspace(settings.db_path)
    legacy.add_dataframe(make_wages(), "wages", "synthetic")
    legacy.checkpoint()
    legacy.close()
    c = TestClient(create_app(settings, with_memory=False))
    books = c.get("/notebooks").json()
    assert [(b["title"], b["n_datasets"]) for b in books] == [("Mi primer cuaderno", 1)]
    assert not settings.db_path.exists()  # moved, not copied
    assert [d["name"] for d in c.get("/datasets").json()] == ["wages"]


def test_notebooks_do_not_mix_datasets_objects_journal_panel_or_chat(two, tmp_path):
    c, a, b = two
    da, db = c.get(prefixed(c, a, "/datasets")).json(), c.get(prefixed(c, b, "/datasets")).json()
    assert [d["name"] for d in da] == ["d"] == [d["name"] for d in db]
    assert da[0]["n_rows"] == 480 and db[0]["n_rows"] == 100

    # The same object name in both notebooks, each fitted on its own dataset.
    fa = c.post(prefixed(c, a, "/tools/ols_fit"), json={"dataset_id": "d", "formula": "lwage ~ educ", "name": "eq01"})
    assert fa.status_code == 200, fa.text
    assert c.get(prefixed(c, b, "/objects")).json() == []
    fb = c.post(prefixed(c, b, "/tools/ols_fit"), json={"dataset_id": "d", "formula": "rate ~ gdp", "name": "eq01"})
    assert fb.status_code == 200, fb.text
    spec_a = c.get(prefixed(c, a, "/objects/eq01")).json()["spec"]["formula"]
    spec_b = c.get(prefixed(c, b, "/objects/eq01")).json()["spec"]["formula"]
    assert (spec_a, spec_b) == ("lwage ~ educ", "rate ~ gdp")

    # Journal and panel are per notebook too.
    assert all("rate" not in str(e) for e in c.get(prefixed(c, a, "/journal")).json())
    assert any(blk.get("result_id") == "eq01" for blk in c.get(prefixed(c, b, "/panel")).json()["blocks"])

    # A conversation in a stays in a (the scripted model fits the dataset 'wages').
    c.post(prefixed(c, a, "/datasets/import"), json={"path": str(tmp_path / "wages.csv"), "name": "wages"})
    assert events(c.post(prefixed(c, a, "/chat"), json={"message": "return to education?"}))[-1]["type"] == "done"
    assert c.get(prefixed(c, a, "/chat/history")).json()
    assert c.get(prefixed(c, b, "/chat/history")).json() == []
    assert c.get(prefixed(c, a, "/panel")).json()["blocks"] != c.get(prefixed(c, b, "/panel")).json()["blocks"]


def test_unprefixed_routes_follow_the_notebook_opened_last(two):
    c, a, b = two
    first = c.get("/notebooks").json()
    assert [n["id"] for n in first if n["current"]] == [b]  # a new notebook becomes the one in use
    assert c.get("/datasets").json()[0]["n_rows"] == 100
    assert c.post(prefixed(c, a, "/open")).json()["current"] is True
    assert c.get("/datasets").json()[0]["n_rows"] == 480
    assert c.get("/notebooks").json()[0]["id"] == a  # the list is ordered by last use


def test_unknown_notebook_and_token(tmp_path):
    c = client(tmp_path, token="s3cret")
    h = {"x-qtrix-token": "s3cret"}
    assert c.get("/notebooks").status_code == 401
    assert c.get("/notebooks/nb-missing/datasets").status_code == 401  # the prefix does not bypass the token
    assert c.get("/notebooks/nb-missing/datasets", headers=h).status_code == 404
    assert c.get("/notebooks/nb-missing", headers=h).status_code == 404
    assert c.patch("/notebooks/nb-missing", json={"title": "x"}, headers=h).status_code == 404


def test_rename_describe_and_list_counts(two):
    c, a, _ = two
    out = c.patch(f"/notebooks/{a}", json={"title": "Salarios 2024", "description": "panel"}).json()
    assert (out["title"], out["description"], out["n_datasets"], out["n_objects"]) == ("Salarios 2024", "panel", 1, 0)
    assert c.get(f"/notebooks/{a}").json()["title"] == "Salarios 2024"


def test_uploads_land_in_the_notebook_folder(two, tmp_path):
    c, a, b = two
    body = make_wages().to_csv(index=False).encode()
    r = c.post(prefixed(c, a, "/datasets/upload"), files={"file": ("extra.csv", body, "text/csv")})
    assert r.status_code == 200, r.text
    assert (tmp_path / "notebooks" / a / "uploads" / "extra.csv").exists()
    assert not (tmp_path / "notebooks" / b / "uploads" / "extra.csv").exists()


def test_duplicate_is_an_independent_copy(two):
    c, a, _ = two
    copy = c.post(f"/notebooks/{a}/duplicate", json={"title": "Copia"}).json()
    assert copy["title"] == "Copia" and copy["n_datasets"] == 1 and copy["id"] != a
    c.delete(prefixed(c, copy["id"], "/datasets/d"))
    assert c.get(prefixed(c, copy["id"], "/datasets")).json() == []
    assert len(c.get(prefixed(c, a, "/datasets")).json()) == 1  # the original is untouched


def test_delete_moves_to_trash_and_the_default_follows(two, tmp_path):
    c, a, b = two
    assert c.delete(f"/notebooks/{b}").json() == {"ok": True}  # b is the default one
    assert [n["id"] for n in c.get("/notebooks").json()] == [a]
    assert c.get("/datasets").json()[0]["n_rows"] == 480  # unprefixed routes moved to a
    assert any(p.name.startswith(b) for p in (tmp_path / "notebooks" / ".trash").iterdir())
    assert c.get(f"/notebooks/{b}").status_code == 404
    assert c.delete(f"/notebooks/{a}").status_code == 200
    books = c.get("/notebooks").json()  # never left without a notebook
    assert len(books) == 1 and books[0]["id"] not in (a, b) and books[0]["n_datasets"] == 0


def test_notebooks_survive_a_restart(two, tmp_path):
    c, a, b = two
    c.post(prefixed(c, a, "/tools/ols_fit"), json={"dataset_id": "d", "formula": "lwage ~ educ", "name": "eq01"})
    c.post(f"/notebooks/{b}/open")
    for ctx in c.app.state.engine._contexts.values():
        ctx.workspace.checkpoint()
    again = client(tmp_path)  # a new engine process on the same folder
    books = {n["id"]: n for n in again.get("/notebooks").json()}
    assert set(books) == {a, b} and books[b]["current"] and books[a]["n_objects"] == 1
    assert again.get(prefixed(again, a, "/objects/eq01")).json()["spec"]["formula"] == "lwage ~ educ"


@pytest.fixture
def live(tmp_path):
    make_wages().to_csv(tmp_path / "wages.csv", index=False)
    make_macro().to_csv(tmp_path / "macro.csv", index=False)
    app = create_app(Settings(home=tmp_path, token="s3cret", local_only=False), with_memory=False)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    engine = app.state.engine
    engine.url = f"http://127.0.0.1:{port}"
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)
    http = httpx.Client(base_url=engine.url, headers={"x-qtrix-token": "s3cret"}, timeout=300)
    yield engine, http, tmp_path
    http.close()
    server.should_exit = True
    thread.join(timeout=30)
    engine.shutdown()


def test_each_notebook_has_its_own_kernel_that_calls_back_into_it(live):
    pytest.importorskip("jupyter_client")
    pytest.importorskip("ipykernel")
    _, http, home = live
    ids = []
    for csv in ("wages.csv", "macro.csv"):
        nid = http.post("/notebooks", json={"title": csv}).json()["id"]
        http.post(f"/notebooks/{nid}/datasets/import", json={"path": str(home / csv), "name": "d"}).raise_for_status()
        ids.append(nid)
    shapes = []
    for nid in ids:
        out = http.post(f"/notebooks/{nid}/code", json={"code": "marker = 'only here'\nqx.page('d').frame().shape[0]"})
        assert out.status_code == 200, out.text
        assert out.json()["ok"], out.json()
        shapes.append(out.json()["outputs"][-1]["data"]["text/plain"])
    assert shapes == ["480", "100"]  # each kernel reads the dataset of its own notebook
    seen = http.post(f"/notebooks/{ids[1]}/code", json={"code": "'marker' in globals()"}).json()
    assert seen["outputs"][-1]["data"]["text/plain"] == "True"
    other = http.post(f"/notebooks/{ids[0]}/kernel/restart")
    assert other.status_code == 200
    again = http.post(f"/notebooks/{ids[1]}/code", json={"code": "marker"}).json()
    assert again["outputs"][-1]["data"]["text/plain"] == "'only here'"  # restarting a does not touch b
