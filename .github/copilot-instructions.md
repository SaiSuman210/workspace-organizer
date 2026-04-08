# Copilot Instructions

This repo is a complete OpenEnv benchmark environment: **Workspace Organizer**.

## Project layout

```
env/
  models.py       # Pydantic models: File, Action, Observation, Reward, StepResult, TaskSolution
  tasks.py        # Task definitions: easy / medium / hard
  env.py          # WorkspaceEnv — reset() / step() / state() / episode_score()
tests/
  test_models.py
  test_tasks.py
  test_env.py
  test_inference.py
  test_openenv.yaml
inference.py      # Baseline inference script (root-level, required name)
openenv.yaml      # OpenEnv metadata
Dockerfile        # python:3.11-slim, entrypoint inference.py
requirements.txt
```

## Key conventions

- **Env vars**: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` — use these exact names.
- **Log format**: `[START]`, `[STEP]`, `[END]` lines to stdout (see `inference.py`).
- **OpenAI client**: all LLM calls go through `openai.OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)`.
- **No real file I/O**: the environment is fully in-memory.
- **Tests**: run with `python -m pytest tests/ -v` — all 105 tests must pass.

## Reward table

| Action outcome | Delta |
|---|---|
| Rename matches solution | +0.3 |
| Move matches solution placement | +0.4 |
| Delete matches expected duplicate | +0.5 |
| Invalid / unrecognized action | −0.2 |
| Delete non-duplicate / unexpected | −0.5 |

Per-step reward is clamped to `[0.0, 1.0]`. Episode score = mean of per-step rewards.
