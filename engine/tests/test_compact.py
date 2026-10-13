"""Compact agent for small local models: only a core of tools goes with each request, the rest is found with
`search_tools`; and the llama.cpp context window is set from the Models screen."""

from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from qtrix_engine.agents.lead import COMPACT_CORE, build_lead_agent
from qtrix_engine.api import create_app
from qtrix_engine.config import Settings


def run_seeing_tools(agent_kwargs: dict, script: list) -> list[set[str]]:
    """Run the lead agent on ``script`` (tool calls, then text) and return the tool names of each request."""
    seen: list[set[str]] = []

    def respond(messages, info: AgentInfo):
        seen.append({t.name for t in info.function_tools})
        step = script[len(seen) - 1] if len(seen) <= len(script) else "fin"
        return ModelResponse(parts=[ToolCallPart(*step) if isinstance(step, tuple) else TextPart(step)])

    agent = build_lead_agent(FunctionModel(respond), skills_library=None, **agent_kwargs)
    asyncio.run(agent.run("hola"))
    return seen


def test_compact_agent_sends_the_core_and_finds_the_rest_on_demand(ctx, macro):
    [full] = run_seeing_tools({}, ["fin"])
    assert {"ts_arima", "iv_fit", "viz_line"} <= full

    first, second = run_seeing_tools({"compact": True}, [("search_tools", {"queries": ["ts_arima"]}), "fin"])
    assert first == COMPACT_CORE & full and "ts_arima" not in first
    assert "ts_arima" in second and first < second


def test_compact_agent_lists_the_hidden_tools_in_its_instructions(ctx):
    instructions = {}

    def respond(messages, info: AgentInfo):
        instructions["text"] = info.instructions
        return ModelResponse(parts=[TextPart("fin")])

    asyncio.run(build_lead_agent(FunctionModel(respond), skills_library=None, compact=True).run("hola"))
    assert "search_tools" in instructions["text"] and "panel_fit" in instructions["text"]


def test_context_window_is_set_from_the_models_screen(tmp_path):
    c = TestClient(create_app(Settings(home=tmp_path, token=None, local_only=False), with_memory=False))
    assert c.get("/localai/status").json()["ctx_size"] == 32768
    assert c.post("/localai/runtime/config", json={"ctx_size": 16384}).json()["ctx_size"] == 16384
    assert '"ctx_size": 16384' in (tmp_path / "settings.json").read_text(encoding="utf-8")
    assert c.post("/localai/runtime/config", json={"ctx_size": 100}).status_code == 422
