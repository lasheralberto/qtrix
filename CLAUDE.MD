# qtrix: guide for AI coding agents

Local-first agentic econometrics workbench. See [README.md](README.md) and [CONTRIBUTING.md](CONTRIBUTING.md).

## Layout

- `engine/`: Python 3.12 backend (FastAPI, PydanticAI agents, MCP tool registry, DuckDB, Jupyter kernel). Package `qtrix_engine`; public API `qtrix` (`import qtrix as qx`).
- `apps/desktop/`: Electron + React + TypeScript UI. All UI strings go through `src/i18n`.
- `docs/adr/`: architecture decisions. `docs/libraries.md`: econometric library catalog.

## Commands

```bash
npm run setup                                      # install engine (uv) + desktop
npm start                                          # run the app
npm test                                           # engine tests
cd engine && uv run ruff check src tests           # lint
cd apps/desktop && npm run typecheck && npm run build
```

## Rules

- Estimators must be validated against a reference implementation, with an explicit tolerance in the test.
- Every bug fix gets a regression test; every new tool gets a registry entry and a test.
- Keep diffs small; match the surrounding style. No hard-coded UI strings.
- Local-first: never add telemetry or send user data off the machine by default.
- Never commit user workspaces (`*.duckdb`), API keys or `.env` files.
