"""Unit tests for task definitions in env/tasks.py.

Validates Requirements 8.4 and 8.5:
- TASKS registry contains "easy", "medium", "hard" keys
- File count minimums: EASY≥5, MEDIUM≥10, HARD≥12
- HARD task has ≥2 duplicate pairs (same summary and size)
- All tasks have a 'root' folder in initial_folders
"""
from __future__ import annotations

from collections import Counter

import pytest

from env.tasks import TASKS, Task
from env.models import File, TaskSolution


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

def test_tasks_registry_has_all_keys():
    """TASKS dict must contain 'easy', 'medium', and 'hard'."""
    assert "easy" in TASKS
    assert "medium" in TASKS
    assert "hard" in TASKS


def test_tasks_registry_values_are_task_instances():
    for key, task in TASKS.items():
        assert isinstance(task, Task), f"TASKS['{key}'] is not a Task instance"


# ---------------------------------------------------------------------------
# Root folder presence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task_name", ["easy", "medium", "hard"])
def test_task_has_root_folder(task_name):
    task = TASKS[task_name]
    assert "root" in task.initial_folders, f"{task_name} task missing 'root' folder"


# ---------------------------------------------------------------------------
# File count minimums (Requirement 8.5)
# ---------------------------------------------------------------------------

def test_easy_file_count():
    assert len(TASKS["easy"].initial_files) >= 5


def test_medium_file_count():
    assert len(TASKS["medium"].initial_files) >= 10


def test_hard_file_count():
    assert len(TASKS["hard"].initial_files) >= 12


# ---------------------------------------------------------------------------
# EASY: rename-only solution (Requirement 8.1)
# ---------------------------------------------------------------------------

def test_easy_solution_rename_only():
    sol = TASKS["easy"].solution
    assert len(sol.expected_renames) >= 1, "EASY solution must have at least one rename"
    assert len(sol.expected_placements) == 0, "EASY solution must not require folder moves"
    assert len(sol.expected_deletions) == 0, "EASY solution must not require deletions"


# ---------------------------------------------------------------------------
# MEDIUM: folder creation + move solution (Requirement 8.2)
# ---------------------------------------------------------------------------

def test_medium_solution_has_placements():
    sol = TASKS["medium"].solution
    assert len(sol.expected_placements) >= 1, "MEDIUM solution must require file moves"


def test_medium_solution_no_deletions():
    sol = TASKS["medium"].solution
    assert len(sol.expected_deletions) == 0, "MEDIUM solution must not require deletions"


# ---------------------------------------------------------------------------
# HARD: ≥2 duplicate pairs + deletions (Requirements 8.3, 8.5)
# ---------------------------------------------------------------------------

def _count_duplicate_pairs(files: list[File]) -> int:
    """Return the number of duplicate pairs (groups of 2+ files with same summary+size)."""
    counter: Counter = Counter((f.summary, f.size) for f in files)
    return sum(1 for count in counter.values() if count >= 2)


def test_hard_has_at_least_two_duplicate_pairs():
    files = TASKS["hard"].initial_files
    pairs = _count_duplicate_pairs(files)
    assert pairs >= 2, f"HARD task must have ≥2 duplicate pairs, found {pairs}"


def test_hard_solution_has_deletions():
    sol = TASKS["hard"].solution
    assert len(sol.expected_deletions) >= 2, "HARD solution must delete at least 2 files"


def test_hard_solution_has_placements():
    sol = TASKS["hard"].solution
    assert len(sol.expected_placements) >= 1, "HARD solution must require folder moves"


def test_hard_deletions_are_actual_duplicates():
    """Every file in expected_deletions must be a duplicate of another file."""
    task = TASKS["hard"]
    file_map = {f.id: f for f in task.initial_files}
    # Build a lookup: (summary, size) -> list of file_ids
    groups: dict[tuple, list[str]] = {}
    for f in task.initial_files:
        key = (f.summary, f.size)
        groups.setdefault(key, []).append(f.id)
    duplicate_ids = {fid for ids in groups.values() if len(ids) >= 2 for fid in ids}

    for fid in task.solution.expected_deletions:
        assert fid in duplicate_ids, (
            f"File '{fid}' in expected_deletions is not a duplicate"
        )


# ---------------------------------------------------------------------------
# Solution file IDs reference actual files
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task_name", ["easy", "medium", "hard"])
def test_solution_file_ids_exist(task_name):
    task = TASKS[task_name]
    file_ids = {f.id for f in task.initial_files}
    sol = task.solution

    for fid in sol.expected_renames:
        assert fid in file_ids, f"Rename target '{fid}' not in {task_name} files"

    for fid in sol.expected_placements:
        assert fid in file_ids, f"Placement target '{fid}' not in {task_name} files"

    for fid in sol.expected_deletions:
        assert fid in file_ids, f"Deletion target '{fid}' not in {task_name} files"
