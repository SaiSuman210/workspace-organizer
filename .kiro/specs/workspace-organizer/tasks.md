# Implementation Plan: Workspace Organizer

## Overview

Implement the Workspace Organizer benchmark environment bottom-up: Pydantic models first, then task definitions, then the environment core, then the inference runner, and finally packaging/tests. Each step integrates with the previous so there is no orphaned code.

## Tasks

- [x] 1. Set up project structure and Pydantic models
  - Create `env/__init__.py`, `env/models.py`, `env/tasks.py`, `env/env.py`
  - Define `File`, `Action`, `Observation`, `Reward`, `StepResult`, `TaskSolution` in `env/models.py`
  - _Requirements: 1.1, 2.4, 9.1_

  - [x] 1.1 Write unit tests for Pydantic schema validation
    - Test that valid model instances serialize/deserialize correctly
    - Test that missing required fields raise `ValidationError`
    - _Requirements: 1.1, 2.4_

- [x] 2. Implement task definitions in `env/tasks.py`
  - Define `Task` dataclass holding `name`, `instruction`, `initial_files`, `initial_folders`, `solution`
  - Implement EASY task: ≥5 files, rename-only solution
  - Implement MEDIUM task: ≥10 files, create_folder + move solution
  - Implement HARD task: ≥12 files including ≥2 duplicate pairs, folder + delete solution
  - Register all three in `TASKS: dict[str, Task]`
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [x] 2.1 Write unit tests for task definitions
    - Verify file counts meet minimums for each task
    - Verify HARD task has ≥2 duplicate pairs (same `summary` and `size`)
    - Verify `TASKS` registry contains `"easy"`, `"medium"`, `"hard"` keys
    - _Requirements: 8.4, 8.5_

- [x] 3. Implement `WorkspaceEnv.reset()` and `state()` in `env/env.py`
  - Initialize internal state: `_files`, `_folders`, `_solution`, `_step_rewards`, `_done`
  - `reset(task_name)` loads from `TASKS`, raises `ValueError` for unknown names, returns `Observation`
  - `state()` returns current `Observation` without mutating state
  - _Requirements: 1.2, 1.3, 2.1, 2.3, 2.5, 9.1, 9.2_

  - [x] 3.1 Write property test for reset produces clean state (Property 2)
    - **Property 2: Reset produces clean state**
    - **Validates: Requirements 2.5**

  - [x] 3.2 Write property test for state() is read-only and idempotent (Property 3)
    - **Property 3: state() is read-only and idempotent**
    - **Validates: Requirements 2.3**

- [x] 4. Implement action handlers and `WorkspaceEnv.step()` in `env/env.py`
  - Implement `rename` handler: validate `file_id` and non-empty `target`, update `_files`
  - Implement `create_folder` handler: validate non-empty unique `target`, update `_folders`
  - Implement `move` handler: validate `file_id`, target folder exists, not already there; update `_folders`
  - Implement `delete` handler: remove file from `_files` and all `_folders`
  - Implement invalid-action fallthrough: unrecognized `action_type` → `−0.2` penalty
  - Wire all handlers into `step()` returning `StepResult`
  - _Requirements: 1.4, 1.5, 3.1, 3.3, 4.1, 4.2, 4.3, 5.1, 5.3, 6.1, 7.5_

  - [x] 4.1 Write property test for invalid reference rejection (Property 1)
    - **Property 1: Invalid reference rejection**
    - **Validates: Requirements 1.4, 1.5**

  - [x] 4.2 Write property test for step() always returns a StepResult (Property 4)
    - **Property 4: step() always returns a StepResult**
    - **Validates: Requirements 2.2**

  - [x] 4.3 Write property test for rename updates file name (Property 5)
    - **Property 5: Rename updates file name**
    - **Validates: Requirements 3.1**

  - [x] 4.4 Write property test for empty/absent target is always invalid (Property 6)
    - **Property 6: Empty or absent target is always invalid**
    - **Validates: Requirements 3.3, 4.3**

  - [x] 4.5 Write property test for create_folder adds a new empty folder (Property 7)
    - **Property 7: create_folder adds a new empty folder**
    - **Validates: Requirements 4.1**

  - [x] 4.6 Write property test for duplicate folder creation is rejected (Property 8)
    - **Property 8: Duplicate folder creation is rejected**
    - **Validates: Requirements 4.2**

  - [x] 4.7 Write property test for move updates folder membership (Property 9)
    - **Property 9: Move updates folder membership**
    - **Validates: Requirements 5.1**

  - [x] 4.8 Write property test for move to same folder is rejected (Property 10)
    - **Property 10: Move to same folder is rejected**
    - **Validates: Requirements 5.3**

  - [x] 4.9 Write property test for delete removes file from all state (Property 11)
    - **Property 11: Delete removes file from all state**
    - **Validates: Requirements 6.1**

  - [x] 4.10 Write property test for invalid action type returns penalty without termination (Property 15)
    - **Property 15: Invalid action type returns penalty without termination**
    - **Validates: Requirements 7.5**

- [x] 5. Implement reward computation in `env/env.py`
  - Add per-step reward accumulation comparing action outcome against `_solution`
  - Clamp per-step reward to `[0.0, 1.0]`
  - Compute episode score as sum of per-step rewards normalized to `[0.0, 1.0]`
  - Apply `−0.5` penalty for unexpected or non-duplicate deletions
  - _Requirements: 6.2, 6.3, 7.1, 7.2, 7.3, 7.4_

  - [x] 5.1 Write property test for per-step reward is bounded (Property 13)
    - **Property 13: Per-step reward is bounded**
    - **Validates: Requirements 7.2**

  - [x] 5.2 Write property test for episode score is bounded and deterministic (Property 14)
    - **Property 14: Episode score is bounded and deterministic**
    - **Validates: Requirements 7.3, 7.4**

  - [x] 5.3 Write property test for unexpected deletion is penalized (Property 12)
    - **Property 12: Unexpected deletion is penalized**
    - **Validates: Requirements 6.3**

  - [x] 5.4 Write unit tests for reward values on solution-matching actions
    - Test rename matching solution yields `+0.3` before clamping
    - Test move matching solution yields `+0.4` before clamping
    - Test delete matching expected duplicate yields `+0.5` before clamping
    - _Requirements: 3.2, 5.2, 6.2_

- [x] 6. Checkpoint — Ensure all environment tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Implement `inference.py`
  - Parse `--task` CLI arg (default `"easy"`)
  - Call `env.reset()` and emit `[START]` JSON log line to stdout
  - Loop: format observation as prompt → call HF router via `openai` client → parse JSON response into `Action` (wrap parse errors as synthetic invalid action) → call `env.step()` → emit `[STEP]` JSON log line
  - Enforce max step budget (50 steps); on budget exhaustion or `done=True` emit `[END]` JSON log line with final episode score
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_

  - [x] 7.1 Write property test for log lines are valid single-line JSON (Property 16)
    - **Property 16: Log lines are valid single-line JSON**
    - **Validates: Requirements 10.2, 10.6**

  - [x] 7.2 Write unit tests for inference runner log format
    - Test `[START]` log contains `task` and `observation` fields
    - Test `[END]` log contains `episode_score` field
    - Test malformed agent JSON produces synthetic invalid action (not a crash)
    - _Requirements: 10.1, 10.3, 10.6_

  - [x] 7.3 Write integration test for full episode with scripted agent
    - Mock HF router; run deterministic action sequence; verify final `[END]` score
    - _Requirements: 10.4, 10.5_

- [x] 8. Create `openenv.yaml` and packaging files
  - Write `openenv.yaml` with `name`, `version`, `description`, `tasks`, `reward_range`
  - Write `requirements.txt` with `pydantic`, `openai`, `hypothesis`, `pytest`, and any other deps
  - Write `Dockerfile`: install from `requirements.txt`, set `inference.py` as entrypoint
  - _Requirements: 11.1, 11.2, 12.1, 12.2, 12.3_

  - [x] 8.1 Write unit test for openenv.yaml validity
    - Parse `openenv.yaml` and assert required fields are present
    - _Requirements: 11.1, 11.2_

- [x] 9. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Property tests use `hypothesis` with `@given` and `settings(max_examples=100)`
- Each property test references its design property number in a comment: `# Feature: workspace-organizer, Property N: ...`
- The inference runner must never crash on bad agent output — always fall back to a synthetic invalid action
- Step budget of 50 prevents runaway episodes on the 2 vCPU / 8 GB target hardware
