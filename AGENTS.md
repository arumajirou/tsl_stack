# Repository Guidelines

## Project Structure & Module Organization
- Core package lives under `src/tsl/` (CLI entry points in `cli/`, data ingestion in `ingest/`, modeling orchestration in `nfops/`, execution pipelines in `runner/`, persistence helpers in `storage/`, and shared utilities in `tools/` and `utils/`). Treat this tree as the single source of truth.
- Tests reside in `tests/` and mirror the CLI and service layers; add new modules alongside matching test files (e.g., `src/tsl/runner/new_feature.py` pairs with `tests/test_runner_new_feature.py`).
- Runtime artifacts land in `nf_auto_runs/`, `lightning_logs/`, and `artifacts/`; keep these directories out of commits. Shared scripts live in `scripts/` (bootstrap via `scripts/dev_env.sh`) and SQL migrations in `db/migrations/`.

## Build, Test, and Development Commands
- `make install` — upgrades pip, installs the package in editable mode, and pulls in dev dependencies (`black`, `ruff`, `pytest`).
- `make test` — executes the full pytest suite (heavy GPU runs included when enabled).
- `make test-fast` — runs quick regression coverage (`cli`, `ingest`, `runner_dry`) for tight feedback.
- `make run-smoke` / `make run-gpu` — launches the automated pipeline on `gpu_smoke.csv` using CPU or GPU settings; great for validating end-to-end wiring before bespoke experiments.
- `make diagnose` — inspects `nf_auto_runs/` for troubleshooting; couple it with `make clean-plan` before nuking artifacts via `make clean-all`.

## Coding Style & Naming Conventions
- Target Python 3.11 with 4-space indentation, comprehensive type hints, and docstrings for user-facing helpers.
- Use `black` for formatting (`python -m black src tests`) and `ruff` for quick lint passes (`python -m ruff check src tests`); keep diffs lint-clean before review.
- Modules and packages use lowercase snake_case (`tsl.runner`), public classes use PascalCase, and functions/variables adopt snake_case; align CLI command names with existing verbs (e.g., `run_auto`).

## Testing Guidelines
- Pytest is the lone test runner; default options skip `@pytest.mark.heavy`. Opt in with `pytest -m heavy` when validating long GPU or end-to-end jobs.
- Mark database touches with `@pytest.mark.db` and integration flows with `@pytest.mark.e2e` to retain clear signal.
- Prefer naming tests after observable behavior (`test_run_auto_saves_model`) and keep fixtures in `tests/conftest.py`.

## Commit & Pull Request Guidelines
- Follow the existing conventional-commit verbs (`chore:`, `feat:`, `fix:`, `docs:`) seen in Git history; keep subjects concise and imperative.
- Before opening a PR, run at least `make test-fast` (or `make test` when touching critical paths) and summarize the results in the description.
- Reference related issues, include reproduction steps or CLI invocations for reviewer context, and attach screenshots/log snippets when touching notebooks, dashboards, or run outputs.

## Runs & Configuration Tips
- Use `scripts/dev_env.sh` to sync the local virtual environment and environment variables; update it if new secrets or paths are required.
- Store experiment settings under `runner/config/` and keep sensitive credentials in `.env` files excluded from version control. When sharing reproducible runs, commit sanitized configs and note the exact CLI command.
