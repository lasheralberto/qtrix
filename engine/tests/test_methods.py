import pytest
from fastapi.testclient import TestClient

from qtrix_engine.api import create_app
from qtrix_engine.config import Settings
from qtrix_engine.methods import AREAS, methods_catalog
from qtrix_engine.tools.registry import all_tools, get_tool

# Analysis namespaces whose public tools the catalog must cover, and the plumbing tools it leaves to the agent.
ANALYSIS = {"data", "ols", "panel", "iv", "ts", "vol", "viz"}
PLUMBING = {"data_list_datasets"}


def test_every_method_points_at_registered_tools():
    for area in AREAS:
        for m in area.methods:
            assert m.tools, m.id
            for name in m.tools:
                spec = get_tool(name)
                assert spec is not None, f"{m.id}: unknown tool {name}"
                assert not spec.internal, f"{m.id}: {name} is internal"


def test_library_functions_exist():
    algorithms = pytest.importorskip("qtrix_engine.contrib.econometrics.econometric_algorithm")
    for area in AREAS:
        for m in area.methods:
            for fn in m.library:
                assert callable(getattr(algorithms, fn, None)), f"{m.id}: {fn} not in the causal library"


def test_every_analysis_tool_is_in_the_catalog():
    covered = {name for area in AREAS for m in area.methods for name in m.tools}
    public = {s.name for s in all_tools(namespaces=ANALYSIS, only_available=False)}
    assert public - covered - PLUMBING == set()


def test_ids_are_unique():
    assert len({a.id for a in AREAS}) == len(AREAS)
    ids = [m.id for a in AREAS for m in a.methods]
    assert len(set(ids)) == len(ids)


def test_structures_come_from_the_tools():
    by_id = {m["id"]: m for a in methods_catalog() for m in a["methods"]}
    assert by_id["ols"]["structures"] is None
    assert by_id["panel_fit"]["structures"] == ["panel", "panel_undated"]
    assert by_id["unit_root"]["structures"] == ["time_series"]
    assert by_id["event_study"]["structures"] == ["panel", "panel_undated"]


def test_methods_endpoint(tmp_path):
    c = TestClient(create_app(Settings(home=tmp_path, token=None, local_only=True), with_memory=False))
    areas = c.get("/methods").json()["areas"]
    assert [a["id"] for a in areas][:2] == ["data", "cross_section"]
    ols = next(m for a in areas for m in a["methods"] if m["id"] == "ols")
    assert ols["available"] and ols["missing"] == [] and "«y»" in ols["prompt"]


def test_tool_help_endpoint(tmp_path):
    c = TestClient(create_app(Settings(home=tmp_path, token=None, local_only=True), with_memory=False))
    tool = c.get("/methods").json()["areas"][1]["methods"][0]["tools"][0]
    info = c.get(f"/tools/{tool}/help").json()
    assert info["name"] == tool and info["summary"]
    assert all({"name", "type", "required", "default", "doc"} <= p.keys() for p in info["params"])
    assert c.get("/tools/nope/help").status_code == 404
