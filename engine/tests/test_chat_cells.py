"""The chat as editable Python cells (ADR 0002 §7.3, E7·08b)."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from qtrix_engine.agents.lead import describe_edit
from qtrix_engine.api import create_app
from qtrix_engine.config import Settings

TAYLOR = {"dataset_id": "wages", "formula": "lwage ~ educ", "name": "eq_taylor"}


def _prompts(messages) -> list[str]:
    return [p.content for m in messages if isinstance(m, ModelRequest) for p in m.parts
            if isinstance(p, UserPromptPart) and isinstance(p.content, str)]


def taylor_model(seen: list[str]) -> FunctionModel:
    """First turn: estimate eq_taylor, then answer. Later turns: comment on the latest version, no tools."""

    async def stream(messages, info: AgentInfo):
        prompts = _prompts(messages)
        seen.append(prompts[-1])
        answered_tool = getattr(messages[-1].parts[-1], "part_kind", "") == "tool-return"
        if len(prompts) == 1 and not answered_tool:
            yield {0: DeltaToolCall(name="ols_fit", json_args=json.dumps(TAYLOR))}
        elif len(prompts) == 1:
            yield "Estimated eq_taylor."
        else:
            yield "Your new version of eq_taylor adds exper."

    return FunctionModel(stream_function=stream)


def events(resp):
    return [json.loads(line[6:]) for line in resp.text.splitlines() if line.startswith("data: ")]


def make_client(tmp_path, wages_csv, seen):
    c = TestClient(create_app(Settings(home=tmp_path, token=None, local_only=False),
                              model_override=taylor_model(seen), with_memory=False))
    c.post("/datasets/import", json={"path": str(wages_csv)})
    return c


def test_edit_a_generated_cell_and_the_agent_comments_the_new_version(tmp_path, wages_csv):
    seen: list[str] = []
    c = make_client(tmp_path, wages_csv, seen)
    evs = events(c.post("/chat", json={"message": "estima una regla de Taylor"}))
    assert evs[0]["type"] == "turn" and evs[0]["after_block"] is None  # empty panel: the turn opens it
    call = next(e for e in evs if e["type"] == "tool_call")
    result = next(e for e in evs if e["type"] == "tool_result")
    assert call["code"] == 'eq_taylor = qx.equation("eq_taylor", page="wages", method="ls", formula="lwage ~ educ")'
    assert result["ok"] and result["seq"] and result["object"] == "eq_taylor" and result["block_id"]
    assert result["code"] == c.get(f"/journal/{result['seq']}").json()["code"]
    n_blocks = len(c.get("/panel").json()["blocks"])

    # The user edits the formula in the cell and re-runs it: a new version, the live table refreshed in place.
    edited = result["code"].replace("lwage ~ educ", "lwage ~ educ + exper")
    out = c.post("/code", json={"code": edited, "edited_from": result["seq"]}).json()
    assert out["ok"] and out["versions"] == [{"name": "eq_taylor", "version": 2}] and out["stale"] == []
    blocks = c.get("/panel").json()["blocks"]
    assert len(blocks) == n_blocks and "exper" in json.dumps(blocks[0]["data"])

    # The next turn tells the agent what the user changed; it comments on the new version.
    evs = events(c.post("/chat", json={"message": "¿qué opinas?"}))
    assert evs[0]["after_block"] == blocks[-1]["id"]  # the next turn sits after the blocks the first one produced
    note = seen[-1]
    assert note.endswith("¿qué opinas?") and f"edited your call [{result['seq']}]" in note
    assert "formula: + exper)" in note and "eq_taylor is now version 2" in note and "do not revert" in note
    assert "adds exper" in "".join(e["delta"] for e in evs if e["type"] == "text")
    # Reported once: a third turn has nothing new to say.
    events(c.post("/chat", json={"message": "gracias"}))
    assert seen[-1] == "gracias"

    history = c.get("/chat/history").json()
    cells = history[1]["cells"]
    assert history[0]["content"] == "estima una regla de Taylor"  # the transcript keeps what the user wrote
    assert cells[0]["seq"] == result["seq"] and cells[0]["object"] == "eq_taylor" and cells[0]["code"]
    assert history[1]["tools"] == ["ols_fit"] and history[1]["turn_id"]
    assert history[0]["after_block"] is None and history[2]["after_block"] == blocks[-1]["id"]


def test_dependants_of_an_edited_cell_are_reported_stale(tmp_path, wages_csv):
    c = make_client(tmp_path, wages_csv, [])
    for name, formula in (("eq_a", "lwage ~ educ"), ("eq_b", "lwage ~ exper")):
        c.post("/tools/ols_fit", json={"dataset_id": "wages", "formula": formula, "name": name})
    c.post("/tools/ols_compare", json={"result_ids": ["eq_a", "eq_b"], "name": "tab_ab"})
    seq = c.get("/journal").json()[0]["seq"]
    out = c.post("/code", json={"code": c.get("/objects/eq_a/code").json()["code"].replace("educ", "educ + exper"),
                                "edited_from": seq}).json()
    assert out["versions"] == [{"name": "eq_a", "version": 2}] and out["stale"] == ["tab_ab"]


def test_live_checks_and_completions(tmp_path, wages_csv):
    c = make_client(tmp_path, wages_csv, [])

    def issues(code):
        return [i["message"] for call in c.post("/code", json={"code": code, "dry_run": True}).json()["calls"]
                for i in call["issues"]]

    assert issues('qx.equation("eq1", page="wages", formula="lwage ~ educ + exper")') == []
    assert issues('qx.equation("eq1", page="wages", formula="lwage ~ educ + gap")') == [
        "Unknown variable 'gap' in page 'wages'"]
    (lag,) = issues('qx.equation("eq1", page="wages", formula="lwage ~ educ(-1)")')
    assert "cross-section" in lag  # wages has no time: lags are not valid there
    assert issues('qx.page("wages").generate("x2", "exper ** 2 + nope")') != []
    assert issues('qx.equation("eq1", page="nope", formula="y ~ x")') == ["No page named 'nope'"]
    for half_typed in ('qx.equation("eq1", formula=', "qx.obj('x').rename(nam='y')"):
        assert c.post("/code", json={"code": half_typed, "dry_run": True}).json()["kind"] == "invalid"
    assert c.post("/code", json={"code": "import os", "dry_run": True}).json()["kind"] == "python"

    names = c.get("/code/completions").json()
    assert "equation" in names["functions"]["qx"] and "generate" in names["functions"]["page"]
    assert "view" in names["functions"]["obj"]
    assert names["pages"][0]["name"] == "wages" and "educ" in names["pages"][0]["columns"]

    # Signature help: every function has its qx arguments, with types, defaults and docs from the tool.
    sigs = names["signatures"]
    assert set(sigs) == {f"{on}.{n}" for on, ns in names["functions"].items() for n in ns}
    eq = {s["method"]: s for s in sigs["qx.equation"]["signatures"]}
    assert {"ls", "wls", "tsls", "arima", "garch"} <= set(eq)
    ls = {p["name"]: p for p in eq["ls"]["params"]}
    assert list(ls)[:4] == ["name", "page", "method", "formula"] and "weights" not in ls
    assert ls["formula"]["required"] and "patsy" in ls["formula"]["doc"]
    assert ls["cov_type"]["default"] == "'HC1'" and "HC3" in ls["cov_type"]["type"]
    assert next(p for p in eq["wls"]["params"] if p["name"] == "weights")["required"]
    tsls = {p["name"]: p for p in eq["tsls"]["params"]}
    assert tsls["method"]["default"] == "'tsls'" and "formula" in tsls  # iv_fit's own method= is fixed, not shown
    gen = sigs["page.generate"]["signatures"][0]
    assert "dataset_id" not in {p["name"] for p in gen["params"]}  # carried by the page handle
    assert sigs["obj.estimate"]["signatures"][0]["params"][-1]["name"] == "**changes"


def test_describe_edit():
    assert describe_edit({"formula": "r ~ infl + gap", "cov_type": "HC1"},
                         {"formula": "r ~ infl + gap(-1) + r(-1)", "cov_type": "HC3"}) == (
        "formula: + gap(-1), + r(-1), - gap; cov_type: 'HC1' → 'HC3'")
    assert describe_edit({"formula": "y ~ x"}, {"formula": "dlog(y) ~ x"}) == "formula: dependent y → dlog(y)"
    assert describe_edit({"a": 1}, {"a": 1}) == ""
