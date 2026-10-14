<div align="center">

[![English](https://img.shields.io/badge/English-active-2f62d6?style=for-the-badge)](README.md)
[![Español](https://img.shields.io/badge/Espa%C3%B1ol-cambiar-lightgrey?style=for-the-badge)](README.es.md)

</div>

# qtrix

**A team of AI econometricians that works on your computer.**

Talk to it in plain language ("estimate a fixed-effects panel and tell me if there is heteroskedasticity") and
qtrix runs real estimators (statsmodels, linearmodels, arch…) and builds a live workspace with regression tables,
tests, charts and notes. **Your data never leaves your machine** (unless you choose a cloud model).

<p align="center">
  <a href="docs/media/brag.mp4">
    <img src="docs/media/brag-preview.gif" alt="qtrix in 27 seconds: you ask a question, the agent works in its workspace, builds tables, charts and tests, and the report writes itself" width="760">
  </a>
  <br><sub>▶ <a href="docs/media/brag.mp4">Watch the full video (27 s, with sound)</a>: the conversation, a zoom into the agent's workspace and its whole process (plan, tasks, tools, findings), the results and the final report.</sub>
</p>

---

## Install in 3 steps

### 1. Install the requirements (one time only)

| Tool | What for | Download |
|---|---|---|
| **Node.js 20+** | Desktop interface | https://nodejs.org |
| **uv** | Manages Python 3.12 and the libraries for you | https://docs.astral.sh/uv/getting-started/installation/ |
| **Git** | Download the project | https://git-scm.com |

> You don't need to install Python: `uv` downloads it automatically.

Check that they are ready:

```bash
node --version    # v20 or higher
uv --version
git --version
```

### 2. Download and install qtrix

```bash
git clone https://github.com/lasheralberto/qtrix.git
cd qtrix
npm run setup
```

`npm run setup` installs everything: the Python engine with the econometrics stack and the desktop app. The first
run takes a few minutes.

### 3. Start it

```bash
npm start
```

The qtrix window opens. Done.

---

## First steps inside the app

1. **Pick a model** in **Settings**. You have two paths:
   - **100% local and free:** open **Local models** (left bar), download one of the recommended models and qtrix
     runs it for you. It also works with [Ollama](https://ollama.com) or LM Studio.
   - **In the cloud:** paste your Anthropic, OpenAI, Gemini or Mistral key (it is stored in the system keychain).
     Turn on **Local models only** if you want to block every remote provider.
2. **Load your data** by dragging a CSV or Excel file into the data bar.
3. **Ask in the chat**, for example: *"Regress wage on education and experience with robust standard errors"*.
4. Watch the central panel fill up with tables, charts and notes. Keep talking and it transforms.

---

## Something not working?

| Symptom | Fix |
|---|---|
| `uv: command not found` | Restart the terminal after installing uv, or check step 1. |
| The Electron window doesn't open | Run `node apps/desktop/node_modules/electron/install.js` and go back to `npm start`. |
| Error from the VS Code terminal | Use `npm start` (it already clears the `ELECTRON_RUN_AS_NODE` variable that breaks Electron). |
| The agent doesn't answer | Check that a model is configured in **Settings** and look at `~/qtrix/logs/engine.log`. |

---

## Where qtrix keeps your stuff

Everything lives in a single folder, `~/qtrix` by default (change it in **Settings > qtrix folder**):

```
qtrix/
  settings.json   data/ (DuckDB)   memory/   models/ (GGUF)   runtime/ (llama.cpp)
  logs/           uploads/         exports/  LEEME.txt
```

---

## For developers

```
apps/desktop   Electron + React (data bar · dynamic panel · chat with agents)
engine         Python engine: FastAPI, PydanticAI agent, MCP tool registry,
               DuckDB, project memory with turbovec
docs           Library catalog and architecture decision records (ADR)
```

**Browser-only interface** (with the engine started by hand):

```bash
cd engine && uv run python -m qtrix_engine --port 8765
cd apps/desktop && npm run dev:web        # http://localhost:5173
```

**Use the tools from any MCP client** (Claude Desktop, etc.):

```bash
cd engine && uv run qtrix-mcp --workspace path/to/workspace.duckdb
```

It exposes the same registry the agents use (`data_*`, `ols_*`, `panel_*`, `iv_*`, `ts_*`, `viz_*`, `ui_*`,
`memory_*`). The running app also accepts `POST /tools/{name}` for direct calls without an LLM.

**Tests:**

```bash
npm test                                           # engine (pytest)
cd engine && uv run ruff check src tests           # lint
cd apps/desktop && npm run typecheck && npm run build
```

Estimates are validated against statsmodels/linearmodels (coefficients and standard errors to 1e-10) and the
chat → tool → panel loop is tested with a simulated model.

**Build an installer** (`.exe`, `.dmg`, `.AppImage`): `cd apps/desktop && npm run dist`.

---

## Contributing

Contributions are welcome! Read [CONTRIBUTING.md](CONTRIBUTING.md), open an issue to discuss your idea, then send a
pull request. New estimators must be validated against a reference implementation. On your first PR a bot will ask
you to sign the [CLA](CLA.md). Security issues: [SECURITY.md](SECURITY.md).

## License

qtrix is **source-available** under the [Business Source License 1.1](LICENSE): you may use, study and modify it
freely (including in production, for research, teaching, internal use or consulting), but you may not offer it to
third parties as a hosted service or competing product. Each version automatically converts to **Apache-2.0** four
years after its release. For other uses, contact the author.
