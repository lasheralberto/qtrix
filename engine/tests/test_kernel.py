"""The project kernel, client-mode qx and the Python-cell bridge, against a live engine (ADR 0002 §7.2-7.4, E7·07b)."""

from __future__ import annotations

import socket
import sys
import threading
import time

import httpx
import pytest
import uvicorn

from qtrix_engine.api import create_app
from qtrix_engine.config import Settings
from qtrix_engine.contrib.econometrics import LIB_DIR
from qtrix_engine.dispatch import Call, PolicyDenied, dispatch
from qtrix_engine.qx.client import Client, EngineError

from .conftest import make_macro, make_wages

pytest.importorskip("jupyter_client")
pytest.importorskip("ipykernel")

TOKEN = "s3cret"
QUANTREG = """import statsmodels.formula.api as smf
import qtrix as qx

df = qx.page("wages").frame()
res = smf.quantreg("lwage ~ educ + exper", df).fit(q=0.5)
qx.register("eq_q50", res, page="wages")
"""


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    home = tmp_path_factory.mktemp("home")
    wages, macro = home / "wages.csv", home / "macro.csv"
    make_wages().to_csv(wages, index=False)
    make_macro().to_csv(macro, index=False)
    app = create_app(Settings(home=home, token=TOKEN, local_only=False), with_memory=False)
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
    http = httpx.Client(base_url=engine.url, headers={"x-qtrix-token": TOKEN}, timeout=300)
    http.post("/datasets/import", json={"path": str(wages)}).raise_for_status()
    http.post("/datasets/import", json={"path": str(macro)}).raise_for_status()
    http.post("/tools/data_set_structure", json={"dataset_id": "macro", "kind": "time_series",
                                                 "date_col": "quarter"}).raise_for_status()
    yield engine, http
    http.close()
    server.should_exit = True
    thread.join(timeout=30)
    engine.ctx.kernels.shutdown()


def cell(http, code: str, **extra):
    resp = http.post("/code", json={"code": code, **extra})
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_python_cells_run_in_the_kernel_and_keep_state(live):
    _, http = live
    out = cell(http, "import numpy as np\nprint('hello')\nx = 41\nnp.arange(4).sum()", cell_id="nb-1")
    assert out["kind"] == "python" and out["ok"] and not out["mutated"]
    assert out["outputs"][0] == {"type": "stream", "name": "stdout", "text": "hello\n"}
    assert out["outputs"][1]["data"]["text/plain"] == "np.int64(6)"
    assert cell(http, "x + 1")["outputs"][0]["data"]["text/plain"] == "42"
    entry = http.get(f"/journal/{out['seq']}").json()
    assert (entry["kind"], entry["tool"], entry["actor"], entry["channel"], entry["cell_id"], entry["mutating"]) == (
        "python", "py_run", "user", "code", "nb-1", False)
    assert entry["code"].startswith("import numpy as np")
    failed = cell(http, "1 / 0")
    assert not failed["ok"] and failed["error"]["ename"] == "ZeroDivisionError"
    assert not http.get(f"/journal/{failed['seq']}").json()["ok"]


def test_quantile_regression_registered_from_a_cell(live):
    engine, http = live
    out = cell(http, QUANTREG, cell_id="nb-2")
    assert out["ok"] and out["mutated"] and len(out["calls"]) == 1, out
    obj = http.get("/objects/eq_q50").json()
    assert obj["method"] == "python" and obj["spec"]["code"] == QUANTREG and obj["spec"]["page"] == "wages"
    educ = next(c for c in obj["summary"]["coefficients"] if c["name"] == "educ")
    assert abs(educ["coef"] - 0.08) < 0.03 and obj["summary"]["formula"] == "lwage ~ educ + exper"
    nested = http.get(f"/journal/{out['calls'][0]}").json()
    assert (nested["tool"], nested["channel"], nested["cell_id"], nested["actor"]) == (
        "py_register", "kernel", "nb-2", "user")
    assert http.get(f"/journal/{out['seq']}").json()["mutating"]  # the cell entry says it changed the workfile
    code = http.get("/objects/eq_q50/code").json()
    assert code["kind"] == "python" and code["code"] == QUANTREG
    assert any(b.get("result_id") == "eq_q50" for b in http.get("/panel").json()["blocks"])
    with pytest.raises(Exception, match="kernel"):
        engine.ctx.objects.fitted("eq_q50")  # the fitted result lives in the kernel

    # Editing the cell re-estimates the object: the new code is re-run in a fresh kernel.
    edited = QUANTREG.replace("q=0.5", "q=0.25")
    new = cell(http, "qx.obj('eq_q50').estimate(code=" + repr(edited) + ")")
    assert new["ok"] and new["calls"][0]["result"]["version"] == 2
    assert http.get("/objects/eq_q50").json()["spec"]["code"] == edited


def test_series_from_and_frames_over_arrow(live):
    engine, http = live
    out = cell(http, "m = qx.page('macro').frame()\n"
                     "qx.series_from('macro', 'gdp_gap', m['gdp'] - m['gdp'].rolling(4).mean())\n"
                     "type(m.index).__name__")
    assert out["ok"], out
    assert out["outputs"][-1]["data"]["text/plain"] == repr(type(engine.ctx.workspace.frame("macro").index).__name__)
    gap = engine.ctx.workspace.frame("macro")["gdp_gap"]
    assert gap.isna().sum() == 3 and len(gap) == 100
    client = Client(engine.url, TOKEN)
    remote, local = client.frame("macro", ["gdp", "rate"], "1995q1 2000q4"), engine.ctx.workspace.frame(
        "macro", ["gdp", "rate"], "1995q1 2000q4")
    assert remote.index.equals(local.index) and remote.equals(local)


def test_replay_reruns_registered_cells(live):
    _, http = live
    report = http.post("/journal/replay").json()
    assert report["ok"], report
    assert http.get("/objects/eq_q50").json()["version"] == 2
    script = http.get("/journal/script").text
    assert "qx.register(\"eq_q50\", res, page=\"wages\")" in script and "py_register" not in script


def test_agents_need_permission_to_run_code(live):
    engine, http = live
    with pytest.raises(PolicyDenied, match="ask them"):
        dispatch(Call("py_run", {"code": "1 + 1"}), "agent:lead", "chat")
    engine.ctx.allow_agent_code = True
    try:
        out = dispatch(Call("py_run", {"code": "qx.obj('eq_q50').lock()"}), "agent:lead", "chat")
        nested = http.get(f"/journal/{out['calls'][0]}").json()
        assert nested["actor"] == "agent:lead" and nested["channel"] == "kernel"
        with pytest.raises(Exception, match="locked"):  # the agent's code is held to the agent's rules
            dispatch(Call("py_run", {"code": "qx.obj('eq_q50').rename('eq_other')"}), "agent:lead", "chat")
    finally:
        engine.ctx.allow_agent_code = False


def test_kernel_credentials_only_work_inside_a_running_cell(live):
    engine, http = live
    stale = httpx.post(f"{engine.url}/qx/call", json={"tool": "data_list_datasets"},
                       headers={"x-qtrix-token": engine.kernel_token})
    assert stale.status_code == 409  # e.g. a thread an agent's cell left behind
    assert httpx.get(f"{engine.url}/datasets", headers={"x-qtrix-token": engine.kernel_token}).status_code == 401
    client = Client(engine.url, TOKEN)  # an external script of the user: runs as the user, channel "client"
    assert {d["name"] for d in client(Call("data_list_datasets"))} == {"wages", "macro"}
    assert http.get("/journal", params={"tool": "data_list_datasets"}).json()[-1]["channel"] == "client"
    with pytest.raises(EngineError, match="Unknown object"):
        client(Call("obj_get", {"name": "nope"}))


def test_timeouts_interrupt_without_losing_the_kernel(live):
    _, http = live
    out = cell(http, "while True:\n    pass", timeout=2)
    assert not out["ok"] and out["timed_out"] and out["error"]["ename"] == "KeyboardInterrupt"
    assert cell(http, "x")["outputs"][0]["data"]["text/plain"] == "41"  # variables survived
    status = http.post("/kernel/interrupt").json()
    assert status["interrupted"] is False and status["alive"]
    # A cell stuck in native code ignores the interrupt (time.sleep on Windows): the kernel is restarted.
    stuck = cell(http, "import time\ntime.sleep(120)", timeout=1)
    assert stuck["timed_out"]
    if sys.platform == "win32":  # elsewhere time.sleep honours the interrupt and no restart is needed
        assert "restarted" in stuck["error"]["evalue"]
    assert cell(http, "1 + 1")["ok"] and not cell(http, "x")["ok"]  # alive again, with a fresh namespace


def test_a_python_cell_written_in_the_panel_shows_its_outputs(live):
    _, http = live
    code = "import qtrix as qx\nqx.page('wages').frame()[['lwage']].describe()"
    block = http.post("/panel/cells", json={"code": code}).json()
    run = block["data"]["run"]
    assert block["kind"] == "python_cell" and block["source"] == "user"
    assert run["kind"] == "python" and run["ok"]
    assert "text/html" in run["outputs"][-1]["data"]  # a DataFrame comes with its HTML table
    again = http.put(f"/panel/blocks/{block['id']}/cell", json={"code": "1 / 0"}).json()["data"]["run"]
    assert not again["ok"] and again["error"]["ename"] == "ZeroDivisionError"


def test_plotly_figures_come_back_as_their_mime_bundle(live):
    _, http = live
    out = cell(http, "import plotly.express as px\npx.histogram(x=[1, 2, 2, 3])")
    assert out["ok"], out.get("error")
    assert any("application/vnd.plotly.v1+json" in o.get("data", {}) for o in out["outputs"])


CAUSAL = """import pandas as pd
import qtrix as qx
from qtrix_engine.contrib.econometrics.econometric_algorithm import (
    Static_Diff_in_Diff_regression, propensity_score_construction,
    propensity_score_visualize_propensity_score_distribution)

p = qx.page("wpanel").frame(["lwage", "exper", "union", "educ"])
t = pd.Index(p.index.get_level_values(1))
p["treated"] = (p.index.get_level_values(0) < 30).astype(int)
p["post"] = (t >= t.unique().sort_values()[4]).astype(int)
res = Static_Diff_in_Diff_regression(p["lwage"], p["treated"], p["post"], p[["exper"]],
                                     entity_effect=True, time_effect=True, cov_type="cluster_entity")
qx.register("eq_did_lib", res, page="wpanel")
ps = propensity_score_construction(p["union"], p[["educ", "exper"]])
propensity_score_visualize_propensity_score_distribution(p["union"], ps)
"""


@pytest.mark.skipif(not LIB_DIR.exists(), reason="eco-skills not fetched: run `npm run setup:skills`")
def test_the_skill_causal_library_runs_in_cells_and_its_results_rebuild(live, tmp_path):
    engine, http = live
    path = tmp_path / "wpanel.csv"
    make_wages().to_csv(path, index=False)
    http.post("/tools/data_import_file", json={"path": str(path), "name": "wpanel"}).raise_for_status()
    http.post("/tools/data_set_structure", json={"dataset_id": "wpanel", "kind": "panel", "date_col": "year",
                                                 "id_col": "id"}).raise_for_status()
    out = cell(http, CAUSAL)
    assert out["ok"], out.get("error")
    assert any("image/png" in o.get("data", {}) for o in out["outputs"])  # the library's Agg backend still displays
    obj = http.get("/objects/eq_did_lib").json()
    assert obj["method"] == "python" and obj["summary"]["model"] == "PanelOLS"
    did = next(c for c in obj["summary"]["coefficients"] if c["name"] == "treatment_group_treated")
    assert abs(did["coef"]) < 0.2  # no effect was simulated
    # The cell alone rebuilds the object in a fresh kernel: the library import works there too.
    again = cell(http, "qx.obj('eq_did_lib').estimate()")
    assert again["ok"], again.get("error")
    assert http.get("/objects/eq_did_lib").json()["version"] == 2
