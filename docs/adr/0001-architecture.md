# ADR 0001 — qtrix architecture

Status: accepted · 2026-10-11 · "Dynamic panel" and "Reproducibility" refined by
[ADR 0002](0002-workfile-objects.md) (workfile and named objects)

## Context

qtrix must be the reference agentic workbench for econometric research: a desktop app the researcher installs,
where data never leaves the machine, an AI team chooses and runs estimators, and the results panel evolves
with the conversation. It must work with local models (Ollama, LM Studio) and, if the user wants, with cloud
models through their own API key.

## Decisions

| Concern | Decision | Why |
|---|---|---|
| Shell | **Electron + React (Vite, TS)** | mature cross-platform installers, rich web charting (Plotly, KaTeX) |
| Compute | **Python engine as a sidecar** (FastAPI on `127.0.0.1`, random port, per-session token) | the econometrics ecosystem is Python; the sidecar is packaged with PyInstaller |
| Data | **DuckDB workspace** (`~/.qtrix/workspace.duckdb`) with a dataset catalog | columnar, fast imports from CSV/Parquet/Excel, SQL for transformations, single local file |
| Agents | **PydanticAI** | typed tools, many providers (local and cloud), streaming events, MCP support |
| Model gateway | **LiteLLM** in-process (`llm/gateway.py`); PydanticAI reaches it through an in-memory httpx transport, also exposed as OpenAI-compatible `/llm/v1` | one place for every model call, cloud or local: keychain keys, local-only enforcement, ordered fallbacks, token/cost log (`logs/llm_usage.jsonl`); price table loaded offline, telemetry off |
| Tools | **One registry → two front doors**: in-process PydanticAI toolset for the agents and **qtrix-mcp** (MCP server) for any external client | the agents and external MCP clients see exactly the same capabilities; no duplicate wrappers |
| Dynamic panel | **Panel spec**: typed blocks (`regression_table`, `plotly_figure`, `dataframe`, `diagnostics`, `markdown`, `equation`) mutated by ops (add/update/remove/pin/reorder) with versioning, undo/redo and SSE broadcast | the panel is a shared, inspectable state agents can curate — not a log of outputs |
| User control | Pinned blocks are read-only for agents; the agent sees the panel and datasets in its instructions every turn | the researcher stays in charge of the narrative |
| Reproducibility | every tool stores a `result_id` and a Python snippet | results are auditable and exportable to scripts/notebooks |
| Memory | **turbovec** `IdMapIndex` + local multilingual embeddings (fastembed ONNX), persisted per project; recall injected into each run and exposed as `memory_search` | compressed, trainless, local semantic memory shared by the agent team |
| Local inference | **Embedded llama.cpp** (`llama-server --jinja`, official `bNNNN` builds installed into `<qtrix>/runtime`) serving GGUF models downloaded from Hugging Face into `<qtrix>/models`; Ollama/LM Studio remain optional providers | no third-party install, no duplicated model storage, OpenAI-compatible tool calling; the server is tied to the engine with a kill-on-close Job Object (Windows) / `PR_SET_PDEATHSIG` (Linux) so it never outlives it |
| Storage location | **One user-visible qtrix folder** (`~/qtrix` default, movable from the app) holding data, memory, models, runtime and logs; a bootstrap pointer `location.json` in the per-user config dir | users can see, back up and relocate everything qtrix keeps |
| Delegation | **A2A** between the lead agent and specialists (next iteration) | open protocol: specialists can be local processes or the user's own external agents |
| Privacy | loopback-only API, token auth, keys in OS keychain, "local-only" mode that refuses remote providers, no telemetry; datasets are referenced by id and never serialised into prompts | data stays on the machine by construction |

## Consequences

- Tool functions are plain Python with docstrings; adding a library means adding one module to the registry.
- Fitted model objects live in process memory (`ResultStore`); persisting sessions requires serialising
  summaries + code (planned with project save/export).
- The embedding model is downloaded once on first run (Hugging Face); this is the only non-LLM network
  access and is documented in the privacy audit.
