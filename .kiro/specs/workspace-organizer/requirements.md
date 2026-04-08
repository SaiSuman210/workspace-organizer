# Requirements Document

## Introduction

The Workspace Organizer is an OpenEnv-style AI agent benchmark environment that simulates organizing a messy "Downloads" folder. The environment is fully simulated in Python with no real file I/O. An AI agent receives a list of files with metadata, a folder structure, and a natural language instruction, then performs a sequence of actions (rename, move, create_folder, delete) to complete one of three graded tasks of increasing difficulty. The environment exposes a standard reset/step/state API, uses Pydantic models for all data exchange, and produces a deterministic reward signal.

## Glossary

- **Environment**: The Python simulation (`env.py`) that manages state, validates actions, and computes rewards.
- **Agent**: The external AI model (Qwen/Qwen2.5-72B-Instruct) that reads observations and emits actions via `inference.py`.
- **File**: A simulated file record with fields: `id`, `name`, `type`, `date`, `summary`, `size`.
- **Folder**: A named container in the simulated workspace that holds file IDs.
- **Observation**: A Pydantic model containing the current file list, folder structure, and task instruction.
- **Action**: A Pydantic model with fields `action_type`, `file_id`, and `target`.
- **Reward**: A Pydantic model carrying a `score` float and an optional `message`.
- **StepResult**: A Pydantic model bundling `observation`, `reward`, `done`, and `info`.
- **Episode**: A single run of one task from `reset()` to a terminal state.
- **Task**: One of three predefined scenarios (EASY, MEDIUM, HARD) defined in `tasks.py`.
- **Duplicate**: Two files that share the same `summary` and `size` values.
- **Inference Runner**: The `inference.py` script that drives the Agent through an Episode and emits structured stdout logs.

---

## Requirements

### Requirement 1: Simulated File System State

**User Story:** As a benchmark operator, I want the environment to maintain a fully in-memory file system state, so that no real disk I/O occurs and the benchmark is portable and safe.

#### Acceptance Criteria

1. THE Environment SHALL represent all files as `File` Pydantic model instances held in memory.
2. THE Environment SHALL represent the folder structure as a dictionary mapping folder names to lists of file IDs.
3. THE Environment SHALL initialize with a `root` folder present in every task.
4. IF a file operation references a `file_id` that does not exist in the current state, THEN THE Environment SHALL reject the action and return a negative reward without modifying state.
5. IF a folder operation references a folder name that does not exist, THEN THE Environment SHALL reject the action and return a negative reward without modifying state.

---

### Requirement 2: Environment API

**User Story:** As a benchmark operator, I want a standard `reset()`, `step()`, and `state()` API, so that the environment integrates with OpenEnv-compatible evaluation harnesses.

#### Acceptance Criteria

1. WHEN `reset(task_name)` is called, THE Environment SHALL load the named task's initial file list, folder structure, and instruction, then return an `Observation`.
2. WHEN `step(action)` is called, THE Environment SHALL apply the action, compute a `Reward`, advance the internal state, and return a `StepResult`.
3. WHEN `state()` is called, THE Environment SHALL return the current `Observation` without modifying any state.
4. THE Environment SHALL accept `Action` instances as the sole input type for `step()`.
5. WHEN `reset()` is called on an in-progress episode, THE Environment SHALL discard all prior state and restart cleanly.

---

### Requirement 3: Action — rename

**User Story:** As an agent, I want to rename a file, so that I can normalize messy file names to a clean convention.

#### Acceptance Criteria

1. WHEN an `Action` with `action_type="rename"`, a valid `file_id`, and a non-empty `target` string is received, THE Environment SHALL update the file's `name` field to the value of `target`.
2. WHEN a rename action produces a name that matches the expected name for that file in the task's solution, THE Environment SHALL add `+0.3` to the step reward.
3. IF the `target` value is empty or absent, THEN THE Environment SHALL treat the action as invalid and apply a `-0.2` penalty.

---

### Requirement 4: Action — create_folder

**User Story:** As an agent, I want to create new folders, so that I can build a meaningful directory structure before moving files.

#### Acceptance Criteria

1. WHEN an `Action` with `action_type="create_folder"` and a non-empty `target` string is received, THE Environment SHALL add a new folder with that name and an empty file list to the folder structure.
2. IF a folder with the given `target` name already exists, THEN THE Environment SHALL treat the action as invalid and apply a `-0.2` penalty.
3. IF the `target` value is empty or absent, THEN THE Environment SHALL treat the action as invalid and apply a `-0.2` penalty.

---

### Requirement 5: Action — move

**User Story:** As an agent, I want to move a file into a folder, so that I can group related files together.

#### Acceptance Criteria

1. WHEN an `Action` with `action_type="move"`, a valid `file_id`, and a valid `target` folder name is received, THE Environment SHALL remove the file ID from its current folder and add it to the target folder.
2. WHEN a move action places a file in the folder specified by the task's solution, THE Environment SHALL add `+0.4` to the step reward.
3. IF the file is already in the target folder, THEN THE Environment SHALL treat the action as invalid and apply a `-0.2` penalty.

---

### Requirement 6: Action — delete

**User Story:** As an agent, I want to delete a file, so that I can remove duplicates and reduce clutter.

#### Acceptance Criteria

1. WHEN an `Action` with `action_type="delete"` and a valid `file_id` is received, THE Environment SHALL remove the file from all folders and from the file list.
2. WHEN a delete action targets a file that is a duplicate (same `summary` and `size` as another file) and the task solution expects its deletion, THE Environment SHALL add `+0.5` to the step reward.
3. IF a delete action targets a file that is not a duplicate or is not expected to be deleted by the task solution, THEN THE Environment SHALL apply a `-0.5` penalty.

---

### Requirement 7: Reward Computation

**User Story:** As a benchmark operator, I want a deterministic, bounded reward signal per step and per episode, so that agent performance is comparable across runs.

#### Acceptance Criteria

1. THE Environment SHALL compute a per-step reward by summing all positive and negative contributions from the action taken in that step.
2. THE Environment SHALL clamp the per-step reward to the range `[0.0, 1.0]`.
3. WHEN an episode ends, THE Environment SHALL compute a terminal episode-level score as the sum of all per-step rewards, normalized to `[0.0, 1.0]`.
4. THE Environment SHALL produce identical reward values for identical action sequences on the same task across multiple runs (deterministic grading).
5. WHEN an action is invalid (unrecognized `action_type` or missing required fields), THE Environment SHALL apply a `-0.2` penalty and return `done=False`.

---

### Requirement 8: Task Definitions

**User Story:** As a benchmark operator, I want three predefined tasks of increasing difficulty, so that agent capability can be evaluated across a range of complexity levels.

#### Acceptance Criteria

1. THE Environment SHALL provide an EASY task whose solution requires only file renames (e.g., normalizing `IMG_001.JPG` to `photo_001.jpg`).
2. THE Environment SHALL provide a MEDIUM task whose solution requires creating folders and moving files grouped by contextual category (e.g., tax documents, travel photos).
3. THE Environment SHALL provide a HARD task whose solution requires both folder organization and deletion of all duplicate files (files sharing identical `summary` and `size`).
4. WHEN `reset("easy")`, `reset("medium")`, or `reset("hard")` is called, THE Environment SHALL load the corresponding task's files, folders, and instruction.
5. THE Environment SHALL include at least 5 files in the EASY task, at least 10 files in the MEDIUM task, and at least 12 files in the HARD task (including at least 2 duplicate pairs).

---

### Requirement 9: Observation and Instruction

**User Story:** As an agent, I want to receive a clear observation with a natural language instruction, so that I can determine what actions to take.

#### Acceptance Criteria

1. WHEN an `Observation` is returned, THE Environment SHALL include the full current file list, the full current folder structure, and the task instruction string.
2. THE Environment SHALL express the task instruction in natural language (e.g., "Clean up messy file names" or "Organize files by project and remove duplicates").
3. WHEN a file is deleted, THE Environment SHALL remove it from the `files` list in all subsequent `Observation` instances.
4. WHEN a file is moved, THE Environment SHALL reflect the updated folder membership in all subsequent `Observation` instances.

---

### Requirement 10: Inference Runner

**User Story:** As a benchmark operator, I want `inference.py` to drive the agent through a full episode and emit structured logs, so that runs can be monitored and scored externally.

#### Acceptance Criteria

1. WHEN an episode begins, THE Inference_Runner SHALL emit a `[START]` log line to stdout containing the task name and initial observation.
2. WHEN each `step()` is executed, THE Inference_Runner SHALL emit a `[STEP]` log line to stdout containing the action taken, the reward received, and the current `done` flag.
3. WHEN an episode ends, THE Inference_Runner SHALL emit an `[END]` log line to stdout containing the final episode score.
4. THE Inference_Runner SHALL call the Agent (Qwen/Qwen2.5-72B-Instruct via OpenAI-compatible HF router) to generate each action from the current observation.
5. THE Inference_Runner SHALL complete a full episode within 20 minutes on hardware with 2 vCPU and 8 GB RAM.
6. THE Inference_Runner SHALL serialize all log lines as valid JSON objects on a single line each.

---

### Requirement 11: openenv.yaml Metadata

**User Story:** As a benchmark operator, I want a machine-readable metadata file, so that the environment can be registered and discovered in the OpenEnv catalog.

#### Acceptance Criteria

1. THE Environment SHALL include an `openenv.yaml` file at the project root containing at minimum: `name`, `version`, `description`, `tasks` (list of task names), and `reward_range`.
2. WHEN the `openenv.yaml` is parsed, THE Environment SHALL produce a valid YAML document with no syntax errors.

---

### Requirement 12: Containerization

**User Story:** As a benchmark operator, I want the environment to run inside a Docker container, so that it is reproducible across machines.

#### Acceptance Criteria

1. THE Environment SHALL include a `Dockerfile` that installs all dependencies from `requirements.txt` and sets `inference.py` as the default entry point.
2. WHEN the Docker image is built and run, THE Inference_Runner SHALL execute without requiring any host file system mounts.
3. THE Environment SHALL list all Python dependencies in `requirements.txt`, including `pydantic` and the OpenAI-compatible client library.
