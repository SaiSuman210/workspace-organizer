"""WorkspaceEnv — the central environment class.

Exposes the OpenEnv-compatible reset / step / state API.
"""
from __future__ import annotations

import copy
from typing import Optional

from env.models import Action, File, Observation, Reward, StepResult, TaskSolution
from env.tasks import TASKS


class WorkspaceEnv:
    """In-memory workspace organizer environment."""

    def __init__(self) -> None:
        self._files: dict[str, File] = {}
        self._folders: dict[str, list[str]] = {}
        self._solution: Optional[TaskSolution] = None
        self._step_rewards: list[float] = []
        self._done: bool = False
        self._instruction: str = ""

    def reset(self, task_name: str) -> Observation:
        """Load a named task and return the initial observation."""
        if task_name not in TASKS:
            raise ValueError(f"Unknown task name: {task_name!r}. Valid names: {list(TASKS.keys())}")

        task = TASKS[task_name]

        # Deep-copy so mutations don't affect the task definition
        self._files = {f.id: f.model_copy(deep=True) for f in task.initial_files}
        self._folders = copy.deepcopy(task.initial_folders)
        self._solution = task.solution
        self._step_rewards = []
        self._done = False
        self._instruction = task.instruction

        return self.state()

    def step(self, action: Action) -> StepResult:
        """Apply an action, compute reward, and return a StepResult."""
        raw_reward: float = 0.0

        if action.action_type == "rename":
            raw_reward = self._handle_rename(action)
        elif action.action_type == "create_folder":
            raw_reward = self._handle_create_folder(action)
        elif action.action_type == "move":
            raw_reward = self._handle_move(action)
        elif action.action_type == "delete":
            raw_reward = self._handle_delete(action)
        else:
            # Unrecognized action type — penalty, state unchanged
            raw_reward = -0.2

        # Clamp per-step reward to [0.0, 1.0]
        clamped = max(0.0, min(1.0, raw_reward))
        self._step_rewards.append(clamped)

        reward = Reward(score=clamped)
        return StepResult(
            observation=self.state(),
            reward=reward,
            done=self._done,
            info={},
        )

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def episode_score(self) -> float:
        """Return the normalized episode score in [0.0, 1.0]."""
        if not self._step_rewards:
            return 0.0
        raw = sum(self._step_rewards) / len(self._step_rewards)
        return max(0.0, min(1.0, raw))

    def _handle_rename(self, action: Action) -> float:
        """Rename a file. Returns raw reward delta."""
        if not action.file_id or action.file_id not in self._files:
            return -0.2
        if not action.target:
            return -0.2
        self._files[action.file_id] = self._files[action.file_id].model_copy(
            update={"name": action.target}
        )
        # Full reward if matches solution, partial credit for any valid rename
        if action.target == self._solution.expected_renames.get(action.file_id):
            return 0.3
        return 0.1  # partial credit for attempting a rename

    def _handle_create_folder(self, action: Action) -> float:
        """Create a new folder. Returns raw reward delta."""
        if not action.target:
            return -0.2
        if action.target in self._folders:
            return -0.2
        self._folders[action.target] = []
        return 0.0

    def _handle_move(self, action: Action) -> float:
        """Move a file to a target folder. Returns raw reward delta."""
        if not action.file_id or action.file_id not in self._files:
            return -0.2
        if not action.target or action.target not in self._folders:
            return -0.2

        # Find current folder
        current_folder: Optional[str] = None
        for folder_name, file_ids in self._folders.items():
            if action.file_id in file_ids:
                current_folder = folder_name
                break

        if current_folder == action.target:
            return -0.2

        # Move: remove from current folder (if found), add to target
        if current_folder is not None:
            self._folders[current_folder].remove(action.file_id)
        self._folders[action.target].append(action.file_id)
        # Full reward if matches solution, partial credit for any valid move
        if action.target == self._solution.expected_placements.get(action.file_id):
            return 0.4
        return 0.1  # partial credit for attempting a move

    def _handle_delete(self, action: Action) -> float:
        """Delete a file. Returns raw reward delta."""
        if not action.file_id or action.file_id not in self._files:
            return -0.2
        del self._files[action.file_id]
        for file_ids in self._folders.values():
            if action.file_id in file_ids:
                file_ids.remove(action.file_id)
        # Check if deletion matches solution
        if action.file_id in self._solution.expected_deletions:
            return 0.5
        return -0.5

    def state(self) -> Observation:
        """Return the current observation without mutating state."""
        return Observation(
            files=list(self._files.values()),
            folders={folder: list(file_ids) for folder, file_ids in self._folders.items()},
            instruction=self._instruction,
        )
