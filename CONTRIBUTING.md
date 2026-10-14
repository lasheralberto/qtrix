# Contributing to qtrix

Thanks for helping build a better econometrics workbench! / ¡Gracias por ayudar a construir una mejor app de econometría!

## Before you start

- **License & CLA.** qtrix is source-available under the [Business Source License 1.1](LICENSE). You must sign the
  [CLA](CLA.md) once; a bot will ask you on your first pull request.
- **Open an issue first** for anything bigger than a small fix, so we can agree on the approach.
- Be kind: see the [Code of Conduct](CODE_OF_CONDUCT.md).

## Development setup

```bash
git clone https://github.com/lasheralberto/qtrix.git
cd qtrix
npm run setup     # Python engine (uv) + desktop app
npm start         # launch the app
```

## Checks that must pass

```bash
npm test                                          # engine tests (pytest)
cd engine && uv run ruff check src tests          # lint
cd apps/desktop && npm run typecheck && npm run build
```

CI runs the same commands on every pull request.

## Guidelines

- **Econometric correctness first.** New estimators must be validated against a reference implementation
  (statsmodels, linearmodels, R, Stata output…) with a test that states the tolerance.
- **Tests with every change.** Bug fix = regression test. New tool/method = registry entry + test.
- **Keep the diff small and focused.** One topic per pull request; match the surrounding style.
- **UI text goes through i18n** (`apps/desktop/src/i18n`): no hard-coded strings; add keys for English and Spanish.
- **Local-first.** Do not add telemetry or send user data anywhere by default.
- Architecture decisions live in [docs/adr](docs/adr); propose a new ADR for significant changes.

## Pull requests

1. Fork and branch from `main`.
2. Make your change with tests, and run the checks above.
3. Open a PR using the template and sign the CLA when prompted.

## Reporting security issues

Do **not** open a public issue. See [SECURITY.md](SECURITY.md).
