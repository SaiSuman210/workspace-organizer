---
title: Workspace Organizer
emoji: 🗂️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Workspace Organizer — OpenEnv Benchmark

An OpenEnv-style AI agent benchmark that simulates organizing a messy "Downloads" folder.
The agent performs file operations (rename, move, create_folder, delete) to complete tasks
of increasing difficulty. Everything runs fully in-memory — no real disk I/O.

## Environment description

The agent receives a list of simulated files with metadata (name, type, date, summary, size)
and a folder structure, along with a natural language instruction. It must issue a sequence
of actions to satisfy the task objective and maximize the episode score.

## Action space

| `action_type` | Required fields | Description |
|---|---|---|
| `rename` | `file_id`, `target` | Rename a file to `target` |
| `move` | `file_id`, `target` | Move a file into folder `target` |
| `create_folder` | `target` | Create a new empty folder named `target` |
| `delete` | `file_id` | Delete a file |

## Observation space

```json
{
  "files": [{"id": "e001", "name": "IMG_001.JPG", "type": "image", "date": "2024-01-10", "summary": "...", "size": 2048000}],
  "folders": {"root": ["e001", "e002"]},
  "instruction": "Clean up the messy file names..."
}
```

## Tasks

| Task | Files | Objective | Difficulty |
|---|---|---|---|
| `easy` | 5 | Rename all files to clean lowercase names | Easy |
| `medium` | 10 | Create category folders and move files into them | Medium |
| `hard` | 12 (incl. 2 duplicate pairs) | Organize into folders and delete all duplicates | Hard |

## Reward

Per-step rewards (clamped to `[0.0, 1.0]`):

- Rename matching solution: **+0.3**
- Move matching solution placement: **+0.4**
- Delete matching expected duplicate: **+0.5**
- Invalid / unrecognized action: **−0.2**
- Delete non-duplicate or unexpected: **−0.5**

Episode score = mean of per-step rewards, normalized to `[0.0, 1.0]`.

## Setup

```bash
pip install -r requirements.txt
```

## Run baseline inference

```bash
export HF_TOKEN=hf_...
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct

python inference.py --task easy
python inference.py --task medium
python inference.py --task hard
```

## Run tests

```bash
python -m pytest tests/ -v
```

## Docker

```bash
docker build -t workspace-organizer .
docker run -e HF_TOKEN=hf_... -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct workspace-organizer --task easy
```

## Baseline scores

Scores depend on the model used. With a capable instruction-following model:

| Task | Expected score range |
|---|---|
| easy | 0.5 – 1.0 |
| medium | 0.3 – 0.8 |
| hard | 0.2 – 0.6 |
