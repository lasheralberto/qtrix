# qtrix

**A team of AI econometricians that works on your computer.**

Talk to it in plain language ("estimate a fixed-effects panel and tell me if there is heteroskedasticity") and
qtrix runs real estimators (statsmodels, linearmodels, arch…) and builds a live workspace with regression tables,
tests, charts and notes. **Your data never leaves your machine** unless you choose a cloud model.

> Spanish is the primary README language: see [README.md](README.md). The app UI is available in English and Spanish.

<p align="center">
  <a href="docs/media/qtrix-demo.mp4">
    <img src="docs/media/qtrix-demo.gif" alt="qtrix in 20 seconds: ask a question, the agent builds tables, charts and tests, and the report writes itself" width="760">
  </a>
  <br><sub>▶ <a href="docs/media/qtrix-demo.mp4">Watch the full video (21 s, with sound)</a>: the conversation, how the agent builds the results, and the final report.</sub>
</p>

## Quick start

Requirements: **Node.js 20+**, **[uv](https://docs.astral.sh/uv/getting-started/installation/)** (it installs Python 3.12 for you) and **Git**.

```bash
git clone https://github.com/lasheralberto/qtrix.git
cd qtrix
npm run setup     # Python engine with the econometrics stack + desktop app
npm start         # opens the qtrix window
```

Then, in **Settings**, pick a model: a fully local one (**Local models**, or Ollama / LM Studio) or a cloud provider
(Anthropic, OpenAI, Gemini, Mistral; keys are stored in the system keychain). Drag a CSV or Excel file into the data
bar and ask questions in the chat.

## Repository layout

```
apps/desktop   Electron + React (data bar · dynamic panel · agent chat)
engine         Python engine: FastAPI, PydanticAI agents, MCP tool registry, DuckDB, project memory
docs           Library catalog and architecture decision records (ADR)
```

Use the tools from any MCP client (Claude Desktop, etc.): `cd engine && uv run qtrix-mcp --workspace path/to/workspace.duckdb`.

Tests: `npm test`, `cd engine && uv run ruff check src tests`, `cd apps/desktop && npm run typecheck && npm run build`.

## Contributing

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md), open an issue to discuss your idea, then send a pull
request. New estimators must be validated against a reference implementation. On your first PR a bot will ask you to sign
the [CLA](CLA.md). Security issues: [SECURITY.md](SECURITY.md).

## License

qtrix is **source-available** under the [Business Source License 1.1](LICENSE). You may use, study and modify it freely,
including in production for research, teaching, internal business use and consulting, but you may not offer it to third
parties as a hosted service or competing product. Each version automatically converts to **Apache-2.0** four years after
its release. For other uses, contact the author.
