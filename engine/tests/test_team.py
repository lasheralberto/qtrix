"""The agent team: the planner hands parts of a request to configured specialists, each works in a conversation of
its own (tagged ``agent`` in the stream), and the lead answers from their conclusions, which also feed the report."""

from __future__ import annotations

import json
import re

import pytest
from pydantic import ValidationError
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from qtrix_engine.agents import reporting as rp
from qtrix_engine.agents.planner import roster_text
from qtrix_engine.agents.skills import list_skills, skill_capabilities
from qtrix_engine.llm.providers import LEAD_ID, ModelProfile, TeamAgentConfig, TeamConfig, UserSettings

from .test_planner import client, events, last_prompt, text

MEMBERS = [{"id": "a1", "name": "Panelista", "emoji": "📊", "behavior": "Experto en datos de panel."},
           {"id": "a2", "name": "Causalista", "behavior": "Identificación causal.", "skills": ["qtrix-econometrics"]}]
TASKS = [{"agent_id": "a1", "task": "Estimar eq_a"}, {"agent_id": "a2", "task": "Estimar eq_b"},
         {"agent_id": "ghost", "task": "Estimar eq_c"}]


def team_model(delegations: list[dict], prompts: list[str]) -> FunctionModel:
    """The lead's planner delegates ``delegations``; each member estimates the object its task names, settles its
    one step and concludes; the lead's answer is a synthesis. Reporter prompts are collected in ``prompts``."""
    progress: dict[str, int] = {}

    def structured(messages, info: AgentInfo):
        out = info.output_tools[0]
        props = out.parameters_json_schema["properties"]
        prompt = last_prompt(messages)
        if "findings" in props:
            ids = re.findall(r"^### Cell (\S+)", prompt, re.M)
            return ModelResponse(parts=[ToolCallPart(out.name, {"findings": [
                {"id": i, "finding": f"Hallazgo de {i}"} for i in ids]})])
        if "sections" in props:
            prompts.append(prompt)
            return ModelResponse(parts=[ToolCallPart(out.name, {"title": "Informe del equipo",
                                                                "conclusions_md": "Conclusión conjunta."})])
        if "[Task from the lead" in prompt:
            return ModelResponse(parts=[ToolCallPart(out.name, {"steps": ["Estimar"], "goal": "Tarea"})])
        # Without delegations the planner leaves the lead a step: with a team, that work still goes to the members.
        return ModelResponse(parts=[ToolCallPart(out.name, {"steps": [] if delegations else ["Estimar"],
                                                            "goal": "Comparar especificaciones",
                                                            "delegations": delegations})])

    async def run(messages, info: AgentInfo):
        prompt = last_prompt(messages)
        if "[Your team has already worked" in prompt:
            yield "Síntesis del equipo."
            return
        m = re.search(r"Estimar (eq_\w+)", prompt)
        if m is None:
            yield "Respuesta del lead."
            return
        name = m[1]
        step = progress.get(name, 0)
        progress[name] = step + 1
        if step == 0:
            args = {"dataset_id": "wages", "formula": "lwage ~ educ", "name": name}
            yield {0: DeltaToolCall(name="ols_fit", json_args=json.dumps(args))}
        elif step == 1:
            yield {0: DeltaToolCall(name="plan_update", json_args=json.dumps({"step": 1}))}
        else:
            yield f"Conclusión de {name}."

    return FunctionModel(structured, stream_function=run)


def with_team(c, members=MEMBERS, enabled=True):
    current = c.get("/settings").json()
    current["team"] = {"enabled": enabled, "agents": members}
    resp = c.put("/settings", json=current)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_the_team_is_saved_with_the_settings_and_bounded(tmp_path, wages_csv):
    c = client(tmp_path, wages_csv, team_model([], []))
    saved = with_team(c)
    assert [a["id"] for a in saved["team"]["agents"]] == ["lead", "a1", "a2"] and saved["team"]["enabled"]
    assert c.get("/settings").json()["team"]["agents"][2]["skills"] == ["qtrix-econometrics"]

    current = c.get("/settings").json()
    current["team"]["agents"] = [{"id": f"m{i}", "name": f"M{i}"} for i in range(10)]  # + the default member
    assert c.put("/settings", json=current).status_code == 422
    current["team"]["agents"] = [{"id": "x", "name": "A"}, {"id": "x", "name": "B"}]
    assert c.put("/settings", json=current).status_code == 422


def test_settings_without_a_team_have_none_and_ids_are_checked():
    assert TeamConfig().active() == []
    assert TeamConfig(enabled=True, agents=[TeamAgentConfig(id="a", name="A", enabled=False)]).active() == []
    with pytest.raises(ValidationError):
        TeamAgentConfig(id="bad id", name="x")


def test_skills_are_listed_and_a_member_gets_only_its_own(tmp_path, wages_csv):
    c = client(tmp_path, wages_csv, team_model([], []))
    names = [s["name"] for s in c.get("/skills").json()]
    assert "qtrix-econometrics" in names and all(s["description"] for s in list_skills())
    assert skill_capabilities(include=[]) == [] and skill_capabilities(include=["nope"]) == []
    caps = skill_capabilities(include=["qtrix-econometrics", "nope"])
    assert any(getattr(cap, "include", None) == frozenset({"qtrix-econometrics"}) for cap in caps)


def test_members_work_in_order_and_the_lead_answers_from_their_conclusions(tmp_path, wages_csv):
    prompts: list[str] = []
    c = client(tmp_path, wages_csv, team_model(TASKS, prompts))
    with_team(c)
    evs = events(c.post("/chat", json={"message": "compara especificaciones"}))

    kinds = [(e["type"], e.get("agent")) for e in evs if e["type"] in ("agent_spawn", "agent_done")]
    # The unknown member the planner named is skipped.
    assert kinds == [("agent_spawn", "a1"), ("agent_done", "a1"), ("agent_spawn", "a2"), ("agent_done", "a2")]
    spawn = next(e for e in evs if e["type"] == "agent_spawn")
    assert spawn["name"] == "Panelista" and spawn["emoji"] == "📊" and spawn["task"] == "Estimar eq_a"
    calls = [(e["agent"], e["args"]["name"]) for e in evs if e["type"] == "tool_call"]
    assert calls == [("a1", "eq_a"), ("a2", "eq_b")]
    # A member's own turn does not leak: one turn, one done, both the lead's.
    assert [e for e in evs if e["type"] in ("turn", "done") and "agent" in e] == []
    assert text([e for e in evs if "agent" not in e]).strip() == "Síntesis del equipo."
    assert [e["content"] for e in evs if e["type"] == "agent_done"] == ["Conclusión de eq_a.", "Conclusión de eq_b."]

    # Each member signs its calls in the journal, under a turn of its own.
    fits = [e for e in c.app.state.engine.ctx.journal.list() if e["tool"] == "ols_fit"]
    assert [e["actor"] for e in fits] == ["agent:a1", "agent:a2"] and len({e["turn_id"] for e in fits}) == 2

    # Reporting was off, yet a team turn brings the report up to date with the members' work.
    assert any(e["type"] == "reporting" and e["status"] == "done" for e in evs)
    assert "#### Team: Panelista — task: Estimar eq_a" in prompts[-1] and "Conclusión de eq_b." in prompts[-1]

    turn = c.get("/chat/history").json()[-1]
    assert [a["id"] for a in turn["agents"]] == ["a1", "a2"] and turn["content"].strip() == "Síntesis del equipo."
    assert turn["agents"][0]["cells"][0]["object"] == "eq_a" and turn["agents"][0]["plan"][0]["status"] == "done"
    assert turn["agents"][0]["ok"] and turn["agents"][1]["content"] == "Conclusión de eq_b."


def test_without_a_team_delegations_are_ignored(tmp_path, wages_csv):
    c = client(tmp_path, wages_csv, team_model(TASKS, []))
    with_team(c, enabled=False)
    evs = events(c.post("/chat", json={"message": "compara"}))
    assert not any(e["type"] == "agent_spawn" or "agent" in e for e in evs)
    assert "agents" not in c.get("/chat/history").json()[-1]


def test_a_member_on_a_model_it_cannot_use_falls_back_to_the_active_one(tmp_path, wages_csv):
    c = client(tmp_path, wages_csv, team_model([], []))
    engine = c.app.state.engine
    engine.user.profiles.append(ModelProfile(name="gguf", provider="llamacpp", model="m.gguf"))
    override, engine.model_override = engine.model_override, None
    try:
        for profile, expected in (("nope", "ya no existe"), ("gguf", "llama.cpp")):
            _, used, warning = engine._member_model(TeamAgentConfig(id="a", name="A", profile=profile))
            assert used.name == engine.user.active().name and expected in warning
        _, _, warning = engine._member_model(TeamAgentConfig(id="a", name="A"))
        assert warning is None
    finally:
        engine.model_override = override


def test_the_planner_sees_each_member_on_one_line():
    roster = roster_text([TeamAgentConfig(**MEMBERS[1])])
    assert roster == "- a2: Causalista — Identificación causal. (skills: qtrix-econometrics)"


def test_the_report_prompt_carries_each_members_work():
    turn = {"question": "q", "content": "síntesis", "turn_id": "t", "cells": [],
            "agents": [{"id": "a1", "name": "Panelista", "task": "t1", "content": "beta 0.1",
                        "cells": [{"name": "ols_fit", "object": "eq_a", "finding": "educ 0.08"}],
                        "scratch": {"notes": [{"section": "issue", "text": "panel corto"}]}}]}
    prompt = rp.reporter_prompt(None, [turn], {})
    assert "#### Team: Panelista — task: t1" in prompt and "Conclusions: beta 0.1" in prompt
    assert "- [issue] panel corto" in prompt and "ols_fit → eq_a: educ 0.08" in prompt
    assert rp.object_names([turn]) == ["eq_a"]


def test_with_a_team_an_unsplit_plan_goes_to_every_member_who_remembers_its_earlier_conclusions(tmp_path, wages_csv):
    seen: list[str] = []
    model = team_model([], [])
    inner = model.stream_function

    async def run(messages, info):
        seen.append(last_prompt(messages))
        async for item in inner(messages, info):
            yield item

    model.stream_function = run
    c = client(tmp_path, wages_csv, model)
    with_team(c)
    # The planner drafts steps but no delegations: the whole request goes to each member, in order.
    evs = events(c.post("/chat", json={"message": "Estimar eq_a"}))
    assert [e["agent"] for e in evs if e["type"] == "agent_spawn"] == ["a1", "a2"]
    assert all(e["task"] == "Estimar eq_a" for e in evs if e["type"] == "agent_spawn")
    assert [a["id"] for a in c.get("/chat/history").json()[-1]["agents"]] == ["a1", "a2"]

    # Next turn: each member is reminded of what it concluded before.
    seen.clear()
    events(c.post("/chat", json={"message": "Estimar eq_b"}))
    prompts = [p for p in seen if "[Task from the lead" in p]
    assert prompts and all("Your conclusions from earlier turns" in p and "Conclusión de eq_a." in p for p in prompts)
    assert not any("Your conclusions from earlier turns" in p for p in seen if "[Task from the lead" not in p)


def test_the_team_always_has_its_default_member_first_and_working():
    team = TeamConfig(enabled=True, agents=[TeamAgentConfig(id="a", name="A"),
                                            TeamAgentConfig(id=LEAD_ID, name="Jefe", enabled=False)])
    assert [a.id for a in team.agents] == [LEAD_ID, "a"] and team.lead().enabled and team.lead().name == "Jefe"
    assert [a.id for a in team.active()] == ["a"]  # the lead is never delegated to: it delegates
    fresh = TeamConfig().lead()
    assert fresh.id == LEAD_ID and "qtrix-econometrics" in fresh.skills


def test_the_super_agents_instructions_become_the_default_members_behaviour(tmp_path, wages_csv, monkeypatch):
    old = UserSettings.model_validate({"agent": {"instructions_extra": "Sé conciso."}})
    assert old.team.lead().behavior == "Sé conciso." and old.agent.instructions_extra == ""

    import qtrix_engine.api as api_mod

    built: list[dict] = []
    real = api_mod.build_lead_agent
    monkeypatch.setattr(api_mod, "build_lead_agent", lambda model, **kw: built.append(kw) or real(model, **kw))
    c = client(tmp_path, wages_csv, team_model([], []))
    current = c.get("/settings").json()
    current["team"]["agents"][0].update(behavior="Responde en verso.", skills=[])
    c.put("/settings", json=current)
    c.app.state.engine.agent()
    assert built[-1]["instructions_extra"] == "Responde en verso." and built[-1]["skills"] == []
