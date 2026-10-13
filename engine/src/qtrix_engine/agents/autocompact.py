"""Context meter and auto-compaction of the chat: how much of the model's window the conversation fills, and how to
fold it into a summary when it gets close to the limit."""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic_ai import Agent, UsageLimits
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from ..llm.providers import ModelProfile, UserSettings

log = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4  # rough estimate when the provider reports no usage

#: Context window by provider when the profile sets none (llama.cpp uses the configured ``ctx_size``).
DEFAULT_WINDOWS = {"ollama": 8192, "lmstudio": 8192, "openai_compatible": 32768, "anthropic": 200_000,
                   "openai": 128_000, "google": 1_000_000, "mistral": 128_000}

SUMMARY_HEADER = "[Summary of the earlier conversation, compacted to free context. Build on it as if you had read it.]"

COMPACTOR_INSTRUCTIONS = """\
You compact the conversation between an applied economist and their econometrics agent so it can go on with less
context. Write a dense summary, in the user's language, that lets the agent continue without the original messages:
- the user's goals, questions and preferences, and decisions they made (including edits to the agent's calls);
- datasets used, their structure and active samples; named objects (equations, tables...) with their specification
  and the key numbers (coefficients, standard errors, test statistics) and conclusions;
- what failed and why, and what is still pending or was promised.
Use short bullet points. Do not invent anything that is not in the conversation.
"""


def context_window(profile: ModelProfile, user: UserSettings) -> int:
    """The context the chat may fill: the model's window, capped by ``agent.max_context`` to bound the cost."""
    if profile.context_window:
        window = profile.context_window
    elif profile.provider == "llamacpp":
        window = user.llamacpp.ctx_size
    else:
        window = DEFAULT_WINDOWS.get(profile.provider, 32768)
    return min(window, user.agent.max_context)


def _estimate(messages: list[ModelMessage]) -> int:
    return len(render(messages, result_chars=None)) // CHARS_PER_TOKEN


def history_tokens(history: list[ModelMessage], *, overhead: int = 0) -> tuple[int, bool]:
    """Tokens the history fills in the model's context, and whether that is an estimate. The last response that
    reports usage counts everything sent with it (instructions and tool schemas included); what came after it is
    estimated. Without one, ``overhead`` stands for the instructions and tool schemas."""
    for i in range(len(history) - 1, -1, -1):
        m = history[i]
        if isinstance(m, ModelResponse) and m.usage.input_tokens:
            return m.usage.input_tokens + m.usage.output_tokens + _estimate(history[i + 1:]), False
    return (_estimate(history) + overhead if history else 0), True


def overhead(history: list[ModelMessage]) -> int:
    """What each request carries besides the messages (instructions, tool schemas), from the last reported usage."""
    for i in range(len(history) - 1, -1, -1):
        m = history[i]
        if isinstance(m, ModelResponse) and m.usage.input_tokens:
            return max(m.usage.input_tokens - _estimate(history[:i]), 0)
    return 0


def build_compactor(model: Any) -> Agent[None, str]:
    return Agent(model, name="compactor", instructions=COMPACTOR_INSTRUCTIONS, retries=1)


def _clip(text: str, limit: int | None) -> str:
    return text if limit is None or len(text) <= limit else text[:limit] + "…"


def render(history: list[ModelMessage], *, result_chars: int | None = 800) -> str:
    """The conversation as plain text for the compactor: prompts, answers, calls and (clipped) results."""
    lines: list[str] = []
    for m in history:
        for p in m.parts:
            if isinstance(p, UserPromptPart) and isinstance(p.content, str):
                lines.append(f"USER: {p.content}")
            elif isinstance(p, TextPart) and p.content.strip():
                lines.append(f"AGENT: {p.content.strip()}")
            elif isinstance(p, ToolCallPart):
                lines.append(f"CALL {p.tool_name}({_clip(json.dumps(p.args_as_dict(), default=str), 400)})")
            elif isinstance(p, ToolReturnPart):
                content = p.content if isinstance(p.content, str) else json.dumps(p.content, default=str)
                lines.append(f"RESULT {p.tool_name}: {_clip(content, result_chars)}")
            elif isinstance(p, RetryPromptPart):
                lines.append(f"ERROR {p.tool_name or ''}: {_clip(str(p.content), 300)}")
    return "\n".join(lines)


def _fit(text: str, max_chars: int) -> str:
    """Keep the start (the goals) and the end (the latest work) of a transcript that does not fit."""
    if len(text) <= max_chars:
        return text
    head = max_chars // 4
    return text[:head] + "\n[…]\n" + text[-(max_chars - head):]


def summary_history(summary: str) -> list[ModelMessage]:
    return [ModelRequest(parts=[UserPromptPart(f"{SUMMARY_HEADER}\n\n{summary.strip()}")]),
            ModelResponse(parts=[TextPart("Understood: I will continue from this summary.")])]


def only_summary(history: list[ModelMessage]) -> bool:
    """Nothing has happened since the last compaction."""
    first = history[0].parts[0] if history and history[0].parts else None
    return (len(history) == 2 and isinstance(first, UserPromptPart) and isinstance(first.content, str)
            and first.content.startswith(SUMMARY_HEADER))


async def compact(compactor: Agent[None, str] | None, history: list[ModelMessage], *,
                  window: int) -> list[ModelMessage]:
    """The history folded into a summary written by ``compactor``; without one (or if it fails), the clipped
    transcript stands in for the summary."""
    budget = max(window // 2, 2048) * CHARS_PER_TOKEN  # what the compactor may read, leaving room to write
    transcript = _fit(render(history), budget)
    summary = ""
    if compactor is not None:
        try:
            result = await compactor.run(f"Conversation to compact:\n\n{transcript}",
                                         usage_limits=UsageLimits(request_limit=2))
            summary = str(result.output).strip()
        except Exception as exc:  # noqa: BLE001 - the clipped transcript still frees context
            log.warning("Compacting the chat with the model failed, keeping a clipped transcript: %s", exc)
    if not summary:
        summary = _fit(render(history, result_chars=300), budget // 2)
    return summary_history(summary)
