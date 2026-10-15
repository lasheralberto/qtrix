"""Pluggable LLM providers: local runtimes (Ollama, LM Studio, any OpenAI-compatible server) or
cloud APIs with the user's own key. API keys live in the OS keychain, never in files or logs."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field, model_validator

ProviderName = Literal[
    "llamacpp", "ollama", "lmstudio", "openai_compatible", "anthropic", "openai", "google", "mistral",
]

log = logging.getLogger(__name__)

KEYRING_SERVICE = "qtrix"
CLOUD_PROVIDERS = ("anthropic", "openai", "google", "mistral")
LOCAL_DEFAULT_URLS = {"ollama": "http://localhost:11434/v1", "lmstudio": "http://localhost:1234/v1"}
API_KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "google": "GEMINI_API_KEY",
               "mistral": "MISTRAL_API_KEY", "openai_compatible": "QTRIX_OPENAI_COMPATIBLE_API_KEY",
               "huggingface": "HF_TOKEN"}


class ModelProfile(BaseModel):
    name: str
    provider: ProviderName
    model: str
    base_url: str | None = None
    temperature: float = 0.2
    #: Tokens the model can hold; None = the provider's default (llama.cpp: the configured ``ctx_size``).
    context_window: int | None = Field(default=None, ge=1024)

    @property
    def is_local(self) -> bool:
        if self.provider in ("llamacpp", "ollama", "lmstudio"):
            return True
        if self.provider == "openai_compatible" and self.base_url:
            return urlparse(self.base_url).hostname in {"localhost", "127.0.0.1", "::1"}
        return False


class AgentConfig(BaseModel):
    """User-tunable configuration of the super-agent."""

    instructions_extra: str = ""
    namespaces: list[str] | None = None
    max_tool_calls: int = 40
    #: Draft a plan before acting on each message (plan-and-execute); off = plain ReAct loop, one model call less.
    planner: bool = True
    #: Let the agent run Python cells in the project kernel without asking (they run with the user's privileges).
    allow_code: bool = False
    #: Fold the conversation into a summary when it fills ``compact_at`` of the model's context window.
    auto_compact: bool = True
    compact_at: float = Field(default=0.8, gt=0, le=1)
    #: Cap on the context the chat may fill, whatever the model's window: every request resends the whole history,
    #: so a lower cap trades the agent's memory for fewer input tokens per call (Ajustes → Contexto).
    max_context: int = Field(default=1_000_000, ge=4096)


MAX_TEAM = 10


class TeamAgentConfig(BaseModel):
    """A member of the agent team: a specialist the lead delegates parts of a request to."""

    #: Stable id: the member acts in the journal as ``agent:<id>``.
    id: str = Field(min_length=1, max_length=40, pattern=r"^[A-Za-z0-9_-]+$")
    name: str = Field(min_length=1, max_length=60)
    emoji: str = Field(default="", max_length=8)
    #: How the member works, added to its instructions.
    behavior: str = ""
    #: Skills (by name) the member may load; empty = none.
    skills: list[str] = Field(default_factory=list)
    namespaces: list[str] | None = None
    #: Model profile (by name) the member runs on; None = the active one.
    profile: str | None = None
    enabled: bool = True


#: The team's default member: the lead econometrician itself, always present and first.
LEAD_ID = "lead"


def default_lead(behavior: str = "") -> TeamAgentConfig:
    """The default member as a new install gets it: every skill of the library."""
    from ..agents.skills import list_skills  # late: the skills module pulls in the agent harness

    return TeamAgentConfig(id=LEAD_ID, name="Lead econometrician", emoji="🎓", behavior=behavior,
                           skills=[s["name"] for s in list_skills()])


class TeamConfig(BaseModel):
    #: Whether the lead may hand work to the other members; the default member always works.
    enabled: bool = False
    agents: list[TeamAgentConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _shape(self) -> TeamConfig:
        ids = [a.id for a in self.agents]
        if len(ids) != len(set(ids)):
            raise ValueError("Team members need distinct ids")
        lead = next((a for a in self.agents if a.id == LEAD_ID), None) or default_lead()
        lead.enabled = True
        self.agents = [lead, *(a for a in self.agents if a.id != LEAD_ID)]
        if len(self.agents) > MAX_TEAM:
            raise ValueError(f"A team has at most {MAX_TEAM} members, the default one included")
        return self

    def lead(self) -> TeamAgentConfig:
        """The default member: the lead econometrician, which answers every turn."""
        return self.agents[0]

    def active(self) -> list[TeamAgentConfig]:
        """The members the lead may delegate to: none while the team is off."""
        return [a for a in self.agents[1:] if a.enabled] if self.enabled else []


class LlamaCppConfig(BaseModel):
    """Embedded llama.cpp runtime: which build to install and how to serve models."""

    variant: Literal["cpu", "vulkan", "cuda"] = "cpu"
    ctx_size: int = 32768  # agent prompts (system + skills + tool schemas) already exceed 15k tokens


class UserSettings(BaseModel):
    profiles: list[ModelProfile] = Field(default_factory=lambda: [
        ModelProfile(name="Local · Ollama", provider="ollama", model="qwen3:8b"),
    ])
    active_profile: str = "Local · Ollama"
    #: Profiles tried in order when the active model fails (cloud ones are skipped in local-only mode).
    fallbacks: list[str] = Field(default_factory=list)
    local_only: bool = False
    agent: AgentConfig = Field(default_factory=AgentConfig)
    llamacpp: LlamaCppConfig = Field(default_factory=LlamaCppConfig)
    team: TeamConfig = Field(default_factory=TeamConfig)

    @model_validator(mode="after")
    def _lead_behaviour(self) -> UserSettings:
        # Settings from before the team kept the lead's instructions as the "super-agent"'s: they become the
        # default member's behaviour, once.
        if self.agent.instructions_extra.strip():
            lead = self.team.lead()
            lead.behavior = lead.behavior or self.agent.instructions_extra
            self.agent.instructions_extra = ""
        return self

    def active(self) -> ModelProfile:
        for p in self.profiles:
            if p.name == self.active_profile:
                return p
        if not self.profiles:
            raise ValueError("No model profiles configured")
        return self.profiles[0]


class SettingsStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> UserSettings:
        if self.path.is_file():
            return UserSettings.model_validate_json(self.path.read_text(encoding="utf-8"))
        return UserSettings()

    def save(self, settings: UserSettings) -> UserSettings:
        self.path.write_text(settings.model_dump_json(indent=2), encoding="utf-8")
        return settings


# ---- API keys -----------------------------------------------------------------------
def get_api_key(provider: str) -> str | None:
    try:
        import keyring

        key = keyring.get_password(KEYRING_SERVICE, provider)
        if key:
            return key
    except Exception:  # noqa: BLE001 - a missing/locked keychain must not break local models
        pass
    env = API_KEY_ENV.get(provider)
    return os.environ.get(env) if env else None


def set_api_key(provider: str, key: str | None) -> None:
    import keyring

    _cloud_cache.pop(provider, None)
    if key:
        keyring.set_password(KEYRING_SERVICE, provider, key)
    else:
        try:
            keyring.delete_password(KEYRING_SERVICE, provider)
        except keyring.errors.PasswordDeleteError:
            pass


def has_api_key(provider: str) -> bool:
    return bool(get_api_key(provider))


# ---- model construction --------------------------------------------------------------
class ProviderBlocked(PermissionError):
    pass


def build_model(profile: ModelProfile, *, gateway: Any, local_only: bool = False) -> Any:
    """Turn a profile into a PydanticAI model. Every model, cloud or local, is served through the
    LiteLLM gateway (``llm/gateway.py``); this only fails fast on what can be checked up front."""
    if local_only and not profile.is_local:
        raise ProviderBlocked(f"'{profile.name}' sends prompts to {profile.provider}; local-only mode is on")
    if profile.provider in CLOUD_PROVIDERS and not get_api_key(profile.provider):
        raise ValueError(f"No API key stored for {profile.provider}")
    if profile.provider == "openai_compatible" and not profile.base_url:
        raise ValueError("openai_compatible profiles need a base_url")
    from .gateway import gateway_model

    return gateway_model(gateway, profile)


async def detect_local_models(timeout: float = 1.5) -> dict[str, list[str]]:
    """Discover models served by local runtimes (Ollama, LM Studio). Only touches localhost."""
    found: dict[str, list[str]] = {}
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            r = await client.get("http://localhost:11434/api/tags")
            found["ollama"] = [m["name"] for m in r.json().get("models", [])]
        except (httpx.HTTPError, json.JSONDecodeError, KeyError):
            found["ollama"] = []
        try:
            r = await client.get("http://localhost:1234/v1/models")
            found["lmstudio"] = [m["id"] for m in r.json().get("data", [])]
        except (httpx.HTTPError, json.JSONDecodeError, KeyError):
            found["lmstudio"] = []
    return found


# Listed models that cannot hold a text chat with tools (speech, images, music, embeddings…).
_NON_CHAT = ("tts", "image", "banana", "lyria", "robotics", "computer-use", "transcribe", "embed",
             "deep-research", "antigravity", "customtools", "audio", "realtime", "whisper", "dall-e",
             "moderation", "search", "ocr", "aqa", "veo", "imagen")
_cloud_cache: dict[str, list[str]] = {}


def _is_chat_model(model: str) -> bool:
    return not any(word in model.lower() for word in _NON_CHAT)


async def _fetch_cloud_models(client: httpx.AsyncClient, provider: str, key: str) -> list[str]:
    if provider == "google":
        r = await client.get("https://generativelanguage.googleapis.com/v1beta/models",
                             params={"pageSize": 1000}, headers={"x-goog-api-key": key})
        r.raise_for_status()
        return [m["name"].removeprefix("models/") for m in r.json().get("models", [])
                if "generateContent" in m.get("supportedGenerationMethods", [])]
    if provider == "anthropic":
        r = await client.get("https://api.anthropic.com/v1/models", params={"limit": 1000},
                             headers={"x-api-key": key, "anthropic-version": "2023-06-01"})
    else:
        base = {"openai": "https://api.openai.com/v1", "mistral": "https://api.mistral.ai/v1"}[provider]
        r = await client.get(f"{base}/models", headers={"Authorization": f"Bearer {key}"})
    r.raise_for_status()
    return [m["id"] for m in r.json().get("data", [])]


async def list_cloud_models(timeout: float = 8.0) -> dict[str, list[str]]:
    """Chat models offered by each cloud provider that has an API key stored.

    Cached per provider until its key changes; a provider that fails to answer lists nothing
    (a profile can still name a model by hand)."""
    found: dict[str, list[str]] = {}
    async with httpx.AsyncClient(timeout=timeout) as client:
        for provider in CLOUD_PROVIDERS:
            key = get_api_key(provider)
            if not key:
                continue
            if provider not in _cloud_cache:
                try:
                    models = await _fetch_cloud_models(client, provider, key)
                except (httpx.HTTPError, json.JSONDecodeError, KeyError) as exc:
                    log.warning("Could not list %s models: %s", provider, type(exc).__name__)
                    continue
                _cloud_cache[provider] = sorted({m for m in models if _is_chat_model(m)})
            found[provider] = _cloud_cache[provider]
    return found
