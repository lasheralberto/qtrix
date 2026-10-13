"""The context meter and auto-compaction of the chat."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from qtrix_engine.agents import autocompact as ac
from qtrix_engine.api import create_app
from qtrix_engine.config import Settings
from qtrix_engine.llm.providers import ModelProfile, UserSettings


def events(resp):
    return [json.loads(line[6:]) for line in resp.text.splitlines() if line.startswith("data: ")]


def prompts(messages) -> list[str]:
    return [p.content for m in messages if isinstance(m, ModelRequest) for p in m.parts
            if isinstance(p, UserPromptPart) and isinstance(p.content, str)]


def chat_model(seen: list[list[str]]) -> FunctionModel:
    """The compactor answers with a summary; the agent records the prompts it was sent and answers at length."""

    def compactor(messages, info: AgentInfo):
        return ModelResponse(parts=[TextPart("- the user asked about wages; eq01 estimated")])

    async def agent(messages, info: AgentInfo):
        seen.append(prompts(messages))
        yield "Respuesta larga " * 400

    return FunctionModel(compactor, stream_function=agent)


def client(tmp_path, seen, **agent):
    c = TestClient(create_app(Settings(home=tmp_path, token=None, local_only=False), model_override=chat_model(seen),
                              with_memory=False))
    c.app.state.engine.user.agent.planner = False
    for key, value in agent.items():
        setattr(c.app.state.engine.user.agent, key, value)
    return c


def test_the_meter_reports_usage_against_the_window(tmp_path):
    c = client(tmp_path, [])
    empty = c.get("/chat/context").json()
    assert empty["used"] == 0 and empty["limit"] == 8192 and empty["compact_at"] == 0.8  # default profile: Ollama
    evs = events(c.post("/chat", json={"message": "hola"}))
    meter = next(e for e in evs if e["type"] == "context")
    assert meter["used"] > 0 and meter["limit"] == 8192 and not meter["estimated"]
    assert c.get("/chat/context").json()["used"] == meter["used"]


def test_a_full_context_is_compacted_before_the_next_turn(tmp_path):
    seen: list[list[str]] = []
    c = client(tmp_path, seen, compact_at=0.01)  # 1 % of 8192 tokens: one long answer fills it
    events(c.post("/chat", json={"message": "estima el retorno de la educación"}))
    evs = events(c.post("/chat", json={"message": "¿y con efectos fijos?"}))
    done = next(e for e in evs if e["type"] == "compacted")
    assert done["before"] > done["used"]
    # The agent sees the summary instead of the earlier turn, then the new question.
    assert seen[-1][0].startswith(ac.SUMMARY_HEADER) and "eq01 estimated" in seen[-1][0]
    assert "estima el retorno de la educación" not in seen[-1] and seen[-1][-1] == "¿y con efectos fijos?"
    # The user still sees the whole conversation, with the point where it was compacted.
    history = c.get("/chat/history").json()
    assert [t["content"] for t in history if t["role"] == "user"] == ["estima el retorno de la educación",
                                                                      "¿y con efectos fijos?"]
    assert history[2].get("compacted") is True and not history[0].get("compacted")


def test_auto_compact_can_be_switched_off_and_run_by_hand(tmp_path):
    c = client(tmp_path, [], compact_at=0.01, auto_compact=False)
    events(c.post("/chat", json={"message": "uno"}))
    evs = events(c.post("/chat", json={"message": "dos"}))
    assert not any(e["type"] == "compacted" for e in evs)
    before = c.get("/chat/context").json()["used"]
    after = c.post("/chat/compact").json()
    assert 0 < after["used"] < before


def test_without_reported_usage_the_tokens_are_estimated():
    history = [ModelRequest(parts=[UserPromptPart("x" * 400)]), ModelResponse(parts=[TextPart("ok")])]
    used, estimated = ac.history_tokens(history)
    assert estimated and used >= 100
    history[1].usage = RequestUsage(input_tokens=1000, output_tokens=20)
    assert ac.history_tokens(history) == (1020, False)


def test_the_window_follows_the_profile():
    user = UserSettings()
    assert ac.context_window(ModelProfile(name="g", provider="google", model="m"), user) == 1_000_000
    assert ac.context_window(ModelProfile(name="a", provider="anthropic", model="m"), user) == 200_000
    # The cap in Ajustes bounds the cost: every request resends the history.
    user.agent.max_context = 64_000
    assert ac.context_window(ModelProfile(name="g", provider="google", model="m"), user) == 64_000
    user.agent.max_context = 1_000_000
    assert ac.context_window(ModelProfile(name="l", provider="llamacpp", model="m"), user) == user.llamacpp.ctx_size
    custom = ModelProfile(name="o", provider="ollama", model="m", context_window=40_000)
    assert ac.context_window(custom, user) == 40_000


async def test_a_failing_compactor_falls_back_to_the_clipped_transcript():
    history = [ModelRequest(parts=[UserPromptPart("pregunta " * 5000)]), ModelResponse(parts=[TextPart("respuesta")])]
    out = await ac.compact(None, history, window=4096)
    summary = out[0].parts[0].content
    assert summary.startswith(ac.SUMMARY_HEADER) and "USER: pregunta" in summary
    assert len(summary) < len("pregunta " * 5000)
