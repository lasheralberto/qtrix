# Security Policy

## Reporting a vulnerability

Please **do not** open a public issue for security problems. Use GitHub's private vulnerability reporting
("Security" tab → "Report a vulnerability") on this repository.

Include steps to reproduce, affected version/commit and the impact you see. We aim to acknowledge reports
within 7 days.

## Scope

qtrix runs a local engine (FastAPI) and executes model-generated Python in a local kernel. Reports about
the local API token, code-execution boundaries, API-key storage and anything that could send user data off the
machine unintentionally are especially welcome.
