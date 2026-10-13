# qtrix

**Local-first agentic workbench for econometrics.** Researchers chat with a team of AI econometricians that
run real estimators (statsmodels, linearmodels, arch, …) on their own machine and build a live analysis
panel — regression tables, specification tests, charts, notes — that keeps transforming as the
conversation goes on. Data never leaves the computer.

```
apps/desktop   Electron + React UI (data rail · dynamic panel · agent chat)
engine         Python engine: FastAPI sidecar, PydanticAI super-agent, qtrix-mcp tool registry,
               DuckDB workspace, turbovec project memory
docs           Library catalogue and architecture decisions
```

## Run it (development)

Requirements: [uv](https://docs.astral.sh/uv/), Node 20+, optionally [Ollama](https://ollama.com) for local models.

```bash
cd engine && uv sync --all-extras          # Python 3.12 env with the econometrics stack
cd ../apps/desktop && npm install
node node_modules/electron/install.js      # only if npm skipped Electron's install script
npm run dev                                # Vite + Electron; Electron spawns the engine
```

Browser-only UI against a manually started engine:

```bash
cd engine && uv run python -m qtrix_engine --port 8765
cd apps/desktop && npm run dev:web          # http://localhost:5173
```

Configure the model in **Ajustes**: Ollama / LM Studio / any OpenAI-compatible local server, or Anthropic,
OpenAI, Gemini, Mistral with your own key (stored in the OS keychain). "Solo modelos locales" blocks every
remote provider.

### Local models without Ollama

**Modelos locales** (left rail) downloads GGUF models from Hugging Face — curated picks with tool calling,
or any repo via search — and runs them with an embedded **llama.cpp** server that qtrix installs once
(CPU, Vulkan or CUDA build). Downloads resume after interruptions and are verified with SHA-256.

### The qtrix folder

Everything qtrix stores lives in one folder, `~/qtrix` by default (an existing `~/.qtrix` is migrated
automatically). Change it in **Ajustes > Carpeta de qtrix**, moving the data or opening another folder.

```
qtrix/
  settings.json  data/ (DuckDB)  memory/ (turbovec)  models/ (GGUF)  runtime/ (llama.cpp)
  logs/ (engine.log, llama-server.log)  uploads/  exports/  LEEME.txt
```

The location is remembered in `%APPDATA%\qtrix\location.json` (macOS `~/Library/Application Support/qtrix`,
Linux `~/.config/qtrix`); `QTRIX_HOME` overrides it for scripts and tests.

## Use the tools from any MCP client

```bash
cd engine && uv run qtrix-mcp --workspace path/to/workspace.duckdb
```

Exposes the same registry the agents use (`data_*`, `ols_*`, `panel_*`, `iv_*`, `ts_*`, `viz_*`, `ui_*`,
`memory_*`). The running app also accepts `POST /tools/{name}` for direct, LLM-free calls.

## Tests

```bash
cd engine && uv run pytest && uv run ruff check src tests
cd apps/desktop && npm run typecheck && npm run build
```

Estimates are checked against statsmodels/linearmodels directly (coefficients and standard errors to 1e-10),
and the chat → tool → panel loop is tested with a scripted model.
