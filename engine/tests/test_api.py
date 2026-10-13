import json

from fastapi.testclient import TestClient
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from qtrix_engine.api import create_app
from qtrix_engine.config import Settings

OLS_ARGS = {"dataset_id": "wages", "formula": "lwage ~ educ + exper"}


def _has_tool_return(messages) -> bool:
    return any(getattr(p, "part_kind", "") == "tool-return" for m in messages for p in m.parts)


def scripted_model() -> FunctionModel:
    """Fake LLM: first asks for an OLS fit, then answers in text."""

    def respond(messages, info: AgentInfo):
        if not _has_tool_return(messages):
            return ModelResponse(parts=[ToolCallPart("ols_fit", OLS_ARGS)])
        return ModelResponse(parts=[TextPart("Each year of education raises wages by about 8%.")])

    async def stream(messages, info: AgentInfo):
        if not _has_tool_return(messages):
            yield {0: DeltaToolCall(name="ols_fit", json_args=json.dumps(OLS_ARGS))}
        else:
            yield "Each year of education "
            yield "raises wages by about 8%."

    return FunctionModel(respond, stream_function=stream)


def client(tmp_path, token=None):
    app = create_app(Settings(home=tmp_path, token=token, local_only=False), model_override=scripted_model(),
                     with_memory=False)
    return TestClient(app)


def events(resp):
    return [json.loads(line[6:]) for line in resp.text.splitlines() if line.startswith("data: ")]


def test_token_required(tmp_path):
    c = client(tmp_path, token="s3cret")
    assert c.get("/health").status_code == 200
    assert c.get("/datasets").status_code == 401
    assert c.get("/datasets", headers={"x-qtrix-token": "s3cret"}).status_code == 200


def test_import_chat_and_panel(tmp_path, wages_csv):
    c = client(tmp_path)
    info = c.post("/datasets/import", json={"path": str(wages_csv)}).json()
    assert info["n_rows"] == 480
    assert c.get(f"/datasets/{info['id']}/preview?limit=5").json()["total_rows"] == 5

    evs = events(c.post("/chat", json={"message": "What is the return to education?"}))
    kinds = [e["type"] for e in evs]
    assert "tool_call" in kinds and kinds[-1] == "done"
    result = next(e for e in evs if e["type"] == "tool_result")
    assert result["ok"] and "educ" in result["content"]
    assert "8%" in "".join(e["delta"] for e in evs if e["type"] == "text")

    panel = c.get("/panel").json()
    assert panel["blocks"][0]["kind"] == "regression_table"
    block_id = panel["blocks"][0]["id"]
    c.post(f"/panel/blocks/{block_id}/pin", json={"pinned": True})
    assert c.get("/panel").json()["blocks"][0]["pinned"] is True
    assert c.post("/panel/undo").json()["changed"] is True
    assert c.get("/chat/history").json()[0]["role"] == "user"


def test_direct_tool_calls(tmp_path, wages_csv):
    c = client(tmp_path)
    c.post("/datasets/import", json={"path": str(wages_csv)})
    out = c.post("/tools/ols_fit", json={"dataset_id": "wages", "formula": "lwage ~ educ"}).json()
    assert out["result_id"] == "eq01"
    assert c.post("/tools/ols_fit", json={"dataset_id": "nope", "formula": "y ~ x"}).status_code == 400
    assert c.post("/tools/rm_rf", json={}).status_code == 404


async def test_missing_local_model_falls_back_to_installed_one(tmp_path, monkeypatch):
    from qtrix_engine import api as api_module

    async def fake_detect():
        return {"ollama": ["llama3.1:latest"], "lmstudio": []}

    monkeypatch.setattr(api_module, "detect_local_models", fake_detect)
    engine = api_module.Engine(Settings(home=tmp_path, token=None, local_only=False), with_memory=False)
    assert engine.user.active().model == "qwen3:8b"
    await engine.ensure_local_model()
    assert engine.user.active().model == "llama3.1:latest"
    assert engine.store.load().active().model == "llama3.1:latest"  # persisted for the next run


def test_local_only_blocks_cloud_profiles(tmp_path):
    c = TestClient(create_app(Settings(home=tmp_path, token=None, local_only=True), with_memory=False))
    s = c.get("/settings").json()
    s["profiles"].append({"name": "cloud", "provider": "anthropic", "model": "claude-sonnet-5-5"})
    s["active_profile"] = "cloud"
    s["local_only"] = False
    assert c.put("/settings", json=s).json()["local_only"] is True
    resp = c.post("/chat", json={"message": "hi"})
    assert resp.status_code == 400 and "local-only" in resp.json()["detail"]


def test_models_lists_cloud_models_unless_local_only(tmp_path, monkeypatch):
    from qtrix_engine import api as api_module

    async def fake_detect():
        return {"ollama": [], "lmstudio": []}

    async def fake_cloud():
        return {"google": ["gemini-2.5-flash"]}

    monkeypatch.setattr(api_module, "detect_local_models", fake_detect)
    monkeypatch.setattr(api_module, "list_cloud_models", fake_cloud)
    assert client(tmp_path).get("/models/local").json()["google"] == ["gemini-2.5-flash"]
    c = TestClient(create_app(Settings(home=tmp_path / "lo", token=None, local_only=True), with_memory=False))
    assert "google" not in c.get("/models/local").json()


def test_cloud_model_filter_drops_non_chat_models():
    from qtrix_engine.llm.providers import _is_chat_model

    assert _is_chat_model("gemini-2.5-pro")
    assert not _is_chat_model("gemini-2.5-flash-preview-tts")
    assert not _is_chat_model("gemini-3-pro-image")
