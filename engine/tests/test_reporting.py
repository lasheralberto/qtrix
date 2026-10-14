"""Reporting mode: the ReportingAgent rewrites a conversation's report after every turn."""

from __future__ import annotations

import json

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

import qtrix_engine.api as api_mod
from qtrix_engine.agents import reporting as rp

from .test_api import events
from .test_persistence import seeded

TURN = {"question": "Retorno a la educación?", "role": "assistant", "turn_id": "t1", "content": "Un 8%.",
        "cells": [{"name": "ols_fit", "object": "eq01", "finding": "educ: 0.08 (0.01)"}]}


def reporter_model(report: dict) -> FunctionModel:
    def respond(messages, info: AgentInfo):
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, json.dumps(report))])

    return FunctionModel(respond)


def test_turns_pair_each_question_with_its_answer():
    transcript = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1", "turn_id": "t1"},
                  {"role": "user", "content": "q2"}]
    assert [(t["question"], t["content"]) for t in rp.turns_of(transcript)] == [("q1", "a1")]


@pytest.mark.asyncio
async def test_update_report_returns_the_models_report_without_unknown_objects():
    report = {"title": "R", "sections": [{"heading": "Estimación", "body_md": "x",
                                         "objects": [{"name": "eq01"}, {"name": "ghost"}]}]}
    out = await rp.update_report(rp.build_reporter(reporter_model(report)), None, [TURN], known={"eq01"})
    assert out.title == "R" and [o.name for o in out.sections[0].objects] == ["eq01"]


@pytest.mark.asyncio
async def test_a_failing_model_keeps_the_previous_report():
    def boom(messages, info):
        raise RuntimeError("down")

    previous = rp.ConversationReport(title="Antes")
    out = await rp.update_report(rp.build_reporter(FunctionModel(boom)), previous, [TURN], known=set())
    assert out is previous


def fake_update(calls):
    async def update(reporter, previous, turns, *, known=None):
        calls.append((previous, len(turns)))
        return rp.ConversationReport(title=f"v{len(calls)}")

    return update


def test_reporting_is_off_by_default_and_a_turn_does_not_touch_the_report(tmp_path, monkeypatch):
    calls: list = []
    monkeypatch.setattr(api_mod, "update_report", fake_update(calls))
    c, nid, base = seeded(tmp_path)
    (th,) = c.get(f"{base}/threads").json()
    assert th["reporting"] is False
    evs = events(c.post(f"{base}/chat", json={"message": "again"}))
    assert not [e for e in evs if e["type"] == "reporting"] and not calls
    assert c.get(f"{base}/reporting").json() == {"enabled": False, "version": 0, "report": None}


def test_reporting_rebuilds_from_every_turn_then_updates_per_turn_and_persists(tmp_path, monkeypatch):
    calls: list = []
    monkeypatch.setattr(api_mod, "update_report", fake_update(calls))
    c, nid, base = seeded(tmp_path)
    tid = c.get(f"{base}/threads").json()[0]["id"]
    assert c.patch(f"{base}/threads/{tid}", json={"reporting": True}).json()["reporting"] is True

    evs = events(c.post(f"{base}/reporting/rebuild"))
    assert [e["status"] for e in evs if e["type"] == "reporting"] == ["running", "done"]
    assert calls[0] == (None, 1)  # from scratch, over the one turn that existed
    assert c.get(f"{base}/reporting").json()["report"]["title"] == "v1"

    evs = events(c.post(f"{base}/chat", json={"message": "and now?"}))
    assert [e["status"] for e in evs if e["type"] == "reporting"] == ["running", "done"]
    assert calls[1][0].title == "v1" and calls[1][1] == 1  # incremental: the report so far + the new turn
    got = c.get(f"{base}/reporting").json()
    assert got["enabled"] and got["version"] == 2 and got["report"]["title"] == "v2"
