"""Single LLM gateway: every model call (cloud or local) goes through LiteLLM here.

PydanticAI talks the OpenAI chat-completions protocol to this gateway through an in-process httpx
transport (no port, no network hop); the gateway resolves the model, injects the API key from the
keychain, enforces local-only mode, falls back to other profiles when a model fails, and records
tokens and cost of every call. The same gateway is exposed at ``/llm/v1`` for external tools.

Model ids are ``<provider>/<model>`` (e.g. ``google/gemini-3.8-flash``, ``ollama/qwen3:8b``).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import httpx

from .providers import LOCAL_DEFAULT_URLS, ModelProfile, UserSettings, get_api_key

log = logging.getLogger(__name__)

# LiteLLM provider prefix for each qtrix provider.
LITELLM_PREFIX = {
    "google": "gemini", "anthropic": "anthropic", "openai": "openai", "mistral": "mistral",
    "ollama": "ollama_chat", "lmstudio": "lm_studio", "openai_compatible": "openai", "llamacpp": "openai",
}
LITELLM_LOCAL_URLS = {"ollama": "http://localhost:11434", "lmstudio": LOCAL_DEFAULT_URLS["lmstudio"]}
GATEWAY_BASE_URL = "http://qtrix-gateway/v1"


class GatewayError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def _litellm():
    # Never download LiteLLM's price table at import: local-only mode must not touch the network.
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    import litellm

    litellm.telemetry = False
    litellm.drop_params = True  # silently drop OpenAI params a provider does not support
    litellm.suppress_debug_info = True
    return litellm


def model_id(profile: ModelProfile) -> str:
    return f"{profile.provider}/{profile.model}"


class LLMGateway:
    def __init__(self, usage_path: Path, settings: Callable[[], UserSettings],
                 runtime_url: Callable[[], str | None]) -> None:
        self.usage_path = usage_path
        self._settings = settings
        self._runtime_url = runtime_url
        self._extra: dict[str, ModelProfile] = {}  # profiles not saved in settings (e.g. "Probar activo")
        self._lock = threading.Lock()

    def warm_up(self) -> None:
        """Import LiteLLM in the background (it takes a few seconds) so the first chat is not slow."""
        threading.Thread(target=_litellm, name="qtrix-litellm", daemon=True).start()

    def register(self, profile: ModelProfile) -> str:
        self._extra[model_id(profile)] = profile
        return model_id(profile)

    # ---- routing ---------------------------------------------------------------------
    def _profile(self, mid: str) -> ModelProfile:
        for p in self._settings().profiles:
            if model_id(p) == mid:
                return p
        if mid in self._extra:
            return self._extra[mid]
        provider, _, model = mid.partition("/")
        if provider not in LITELLM_PREFIX or not model:
            raise GatewayError(404, f"Modelo desconocido '{mid}': usa <proveedor>/<modelo>")
        return ModelProfile(name=mid, provider=provider, model=model)  # type: ignore[arg-type]

    def route(self, profile: ModelProfile) -> dict[str, Any]:
        """LiteLLM kwargs (model, api_key, api_base) for a profile."""
        if self._settings().local_only and not profile.is_local:
            raise GatewayError(403, f"'{profile.name}' sends prompts to {profile.provider}; local-only mode is on")
        prefix = LITELLM_PREFIX[profile.provider]
        if profile.provider == "llamacpp":
            url = self._runtime_url()
            if not url:
                raise GatewayError(503, "El modelo de llama.cpp no está cargado")
            alias = Path(profile.model).name.removesuffix(".gguf")
            return {"model": f"{prefix}/{alias}", "api_base": url, "api_key": "qtrix-local"}
        if profile.provider in ("ollama", "lmstudio", "openai_compatible"):
            base = profile.base_url or LITELLM_LOCAL_URLS.get(profile.provider)
            if not base:
                raise GatewayError(400, "openai_compatible profiles need a base_url")
            key = get_api_key("openai_compatible") if profile.provider == "openai_compatible" else None
            return {"model": f"{prefix}/{profile.model}", "api_base": base, "api_key": key or "not-needed"}
        key = get_api_key(profile.provider)
        if not key:
            raise GatewayError(401, f"No API key stored for {profile.provider}")
        return {"model": f"{prefix}/{profile.model}", "api_key": key}

    def chain(self, mid: str) -> list[ModelProfile]:
        """The requested model followed by the fallback profiles allowed to replace it."""
        settings = self._settings()
        chain = [self._profile(mid)]
        by_name = {p.name: p for p in settings.profiles}
        for name in settings.fallbacks:
            p = by_name.get(name)
            if p and model_id(p) not in {model_id(c) for c in chain} and (p.is_local or not settings.local_only):
                chain.append(p)
        return chain

    # ---- completion ------------------------------------------------------------------
    async def chat(self, body: dict[str, Any]) -> dict[str, Any] | AsyncIterator[bytes]:
        """OpenAI chat-completions request in; JSON response or SSE byte stream out."""
        litellm = _litellm()
        mid = body.get("model") or ""
        params = {k: v for k, v in body.items() if k not in ("model", "user")}
        stream = bool(params.get("stream"))
        if stream:
            params["stream_options"] = {"include_usage": True}
        chain = self.chain(mid)
        errors: list[str] = []
        for profile in chain:
            try:
                kwargs = self.route(profile)
            except GatewayError as exc:
                if len(chain) == 1:
                    raise
                errors.append(f"{model_id(profile)}: {exc}")
                continue
            started = time.monotonic()
            try:
                response = await litellm.acompletion(**params, **kwargs)
                if not stream:
                    self._record(profile, mid, started, response.usage, kwargs["model"])
                    return response.model_dump(exclude_none=True)
                iterator = response.__aiter__()
                first = await iterator.__anext__()  # provider errors surface here: still time to fall back
            except Exception as exc:  # noqa: BLE001 - any provider failure moves on to the next fallback
                log.warning("LLM %s failed: %s", model_id(profile), exc)
                errors.append(f"{model_id(profile)}: {type(exc).__name__}: {exc}")
                continue
            return self._sse(profile, mid, started, kwargs["model"], first, iterator)
        raise GatewayError(502, " | ".join(errors) or "Ningún modelo disponible")

    async def _sse(self, profile: ModelProfile, mid: str, started: float, litellm_model: str,
                   first: Any, rest: AsyncIterator[Any]) -> AsyncIterator[bytes]:
        usage = None
        chunk = first
        try:
            while True:
                usage = getattr(chunk, "usage", None) or usage
                yield b"data: " + chunk.model_dump_json(exclude_none=True).encode() + b"\n\n"
                try:
                    chunk = await rest.__anext__()
                except StopAsyncIteration:
                    break
        finally:
            # Before [DONE]: the client may close the stream as soon as it sees it.
            self._record(profile, mid, started, usage, litellm_model)
        yield b"data: [DONE]\n\n"

    # ---- usage -----------------------------------------------------------------------
    def _record(self, profile: ModelProfile, requested: str, started: float, usage: Any, litellm_model: str) -> None:
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        cost: float | None = 0.0
        if not profile.is_local:
            try:
                p_cost, c_cost = _litellm().cost_per_token(model=litellm_model, prompt_tokens=prompt,
                                                           completion_tokens=completion)
                cost = p_cost + c_cost
            except Exception:  # noqa: BLE001 - LiteLLM does not know this model's price
                cost = None
        entry = {"ts": time.time(), "model": model_id(profile), "requested": requested,
                 "fallback": model_id(profile) != requested, "prompt_tokens": prompt,
                 "completion_tokens": completion, "cost_usd": None if cost is None else round(cost, 6),
                 "seconds": round(time.monotonic() - started, 2)}
        with self._lock, self.usage_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def usage(self) -> dict[str, Any]:
        per_model: dict[str, dict[str, float]] = {}
        if self.usage_path.is_file():
            for line in self.usage_path.read_text(encoding="utf-8").splitlines():
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                m = per_model.setdefault(e["model"], {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                                      "cost_usd": 0.0, "unpriced_calls": 0, "fallbacks": 0})
                m["calls"] += 1
                m["prompt_tokens"] += e["prompt_tokens"]
                m["completion_tokens"] += e["completion_tokens"]
                if e["cost_usd"] is None:
                    m["unpriced_calls"] += 1
                else:
                    m["cost_usd"] = round(m["cost_usd"] + e["cost_usd"], 6)
                m["fallbacks"] += int(e.get("fallback", False))
        return {"models": per_model, "total_cost_usd": round(sum(m["cost_usd"] for m in per_model.values()), 6)}


class GatewayTransport(httpx.AsyncBaseTransport):
    """Lets the OpenAI client inside PydanticAI call the gateway in-process, with real streaming."""

    def __init__(self, gateway: LLMGateway) -> None:
        self.gateway = gateway

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if not request.url.path.endswith("/chat/completions"):
            return httpx.Response(404, json={"error": {"message": f"Unsupported path: {request.url.path}"}})
        body = json.loads(await request.aread())
        try:
            out = await self.gateway.chat(body)
        except GatewayError as exc:
            return httpx.Response(exc.status, json={"error": {"message": str(exc), "code": exc.status}})
        if isinstance(out, dict):
            return httpx.Response(200, json=out)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=_ByteStream(out))


class _ByteStream(httpx.AsyncByteStream):
    def __init__(self, chunks: AsyncIterator[bytes]) -> None:
        self.chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self.chunks:
            yield chunk


def gateway_model(gateway: LLMGateway, profile: ModelProfile) -> Any:
    """A PydanticAI model whose requests all go through the gateway."""
    from openai import AsyncOpenAI
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.litellm import LiteLLMProvider

    client = AsyncOpenAI(base_url=GATEWAY_BASE_URL, api_key="qtrix", max_retries=0,
                         http_client=httpx.AsyncClient(transport=GatewayTransport(gateway), timeout=600))
    return OpenAIChatModel(gateway.register(profile), provider=LiteLLMProvider(openai_client=client))
