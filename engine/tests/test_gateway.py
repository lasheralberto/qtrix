"""LiteLLM gateway: routing, local-only, fallbacks, usage log and the /llm/v1 endpoint (LiteLLM mocked)."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from qtrix_engine.api import create_app
from qtrix_engine.config import Settings
from qtrix_engine.llm import gateway as gw_module
from qtrix_engine.llm.gateway import GatewayError, LLMGateway
from qtrix_engine.llm.providers import ModelProfile, UserSettings


class FakeResponse(SimpleNamespace):
    def model_dump(self, **_):
        return {"id": "x", "object": "chat.completion", "model": self.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}


@pytest.fixture
def calls(monkeypatch):
    seen: list[dict] = []

    async def acompletion(**kwargs):
        seen.append(kwargs)
        if "broken" in kwargs["model"]:
            raise RuntimeError("model down")
        return FakeResponse(model=kwargs["model"], usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2))

    fake = SimpleNamespace(acompletion=acompletion, cost_per_token=lambda **_: (0.001, 0.002))
    monkeypatch.setattr(gw_module, "_litellm", lambda: fake)
    monkeypatch.setattr(gw_module, "get_api_key", lambda provider: f"key-{provider}")
    return seen


def make(tmp_path, **kw):
    s = UserSettings(profiles=[ModelProfile(name="G", provider="google", model="gemini-3.8-flash"),
                               ModelProfile(name="B", provider="google", model="broken"),
                               ModelProfile(name="L", provider="ollama", model="qwen3:8b")], active_profile="G", **kw)
    return s, LLMGateway(tmp_path / "usage.jsonl", settings=lambda: s, runtime_url=lambda: None)


async def test_routes_cloud_and_local_models(tmp_path, calls):
    _, gw = make(tmp_path)
    await gw.chat({"model": "google/gemini-3.8-flash", "messages": []})
    await gw.chat({"model": "ollama/qwen3:8b", "messages": []})
    assert calls[0]["model"] == "gemini/gemini-3.8-flash" and calls[0]["api_key"] == "key-google"
    assert calls[1]["model"] == "ollama_chat/qwen3:8b" and calls[1]["api_base"] == "http://localhost:11434"


async def test_falls_back_and_records_usage(tmp_path, calls):
    _, gw = make(tmp_path, fallbacks=["G"])
    out = await gw.chat({"model": "google/broken", "messages": []})
    assert out["model"] == "gemini/gemini-3.8-flash"
    usage = gw.usage()["models"]["google/gemini-3.8-flash"]
    assert usage == {"calls": 1, "prompt_tokens": 10, "completion_tokens": 2, "cost_usd": 0.003,
                     "unpriced_calls": 0, "fallbacks": 1}


async def test_local_only_blocks_cloud_and_skips_cloud_fallbacks(tmp_path, calls):
    _, gw = make(tmp_path, local_only=True, fallbacks=["G"])
    with pytest.raises(GatewayError) as err:
        await gw.chat({"model": "google/gemini-3.8-flash", "messages": []})
    assert err.value.status == 403 and not calls
    assert [p.name for p in gw.chain("ollama/qwen3:8b")] == ["L"]


def test_openai_compatible_endpoint_accepts_bearer_token(tmp_path, calls):
    c = TestClient(create_app(Settings(home=tmp_path, token="t", local_only=False), with_memory=False))
    body = {"model": "google/gemini-3.8-flash", "messages": [{"role": "user", "content": "hi"}]}
    assert c.post("/llm/v1/chat/completions", json=body).status_code == 401
    r = c.post("/llm/v1/chat/completions", json=body, headers={"Authorization": "Bearer t"})
    assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "ok"
    assert c.get("/llm/usage", headers={"x-qtrix-token": "t"}).json()["models"]["google/gemini-3.8-flash"]["calls"] == 1
