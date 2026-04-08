"""Unit and property-based tests for WorkspaceEnv.

Tests cover reset(), state(), and related correctness properties.
"""
from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from env.env import WorkspaceEnv
from env.models import Action, Observation, StepResult
from env.tasks import TASKS


# ---------------------------------------------------------------------------
# Unit tests for reset()
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_returns_observation(self):
        env = WorkspaceEnv()
        obs = env.reset("easy")
        assert isinstance(obs, Observation)

    @pytest.mark.parametrize("task_name", ["easy", "medium", "hard"])
    def test_reset_loads_correct_file_count(self, task_name):
        env = WorkspaceEnv()
        obs = env.reset(task_name)
        expected = len(TASKS[task_name].initial_files)
        assert len(obs.files) == expected

    @pytest.mark.parametrize("task_name", ["easy", "medium", "hard"])
    def test_reset_has_root_folder(self, task_name):
        env = WorkspaceEnv()
        obs = env.reset(task_name)
        assert "root" in obs.folders

    @pytest.mark.parametrize("task_name", ["easy", "medium", "hard"])
    def test_reset_instruction_matches_task(self, task_name):
        env = WorkspaceEnv()
        obs = env.reset(task_name)
        assert obs.instruction == TASKS[task_name].instruction

    def test_reset_unknown_task_raises_value_error(self):
        env = WorkspaceEnv()
        with pytest.raises(ValueError, match="Unknown task name"):
            env.reset("nonexistent")

    def test_reset_does_not_mutate_task_definition(self):
        """Mutations to env state must not affect the original task definition."""
        env = WorkspaceEnv()
        env.reset("easy")
        original_name = TASKS["easy"].initial_files[0].name
        # Mutate the env's internal file
        first_id = list(env._files.keys())[0]
        env._files[first_id] = env._files[first_id].model_copy(update={"name": "mutated.jpg"})
        # Task definition must be unchanged
        assert TASKS["easy"].initial_files[0].name == original_name

    def test_reset_clears_prior_state(self):
        """Calling reset() on an in-progress episode discards all prior state."""
        env = WorkspaceEnv()
        env.reset("easy")
        # Simulate some state mutation
        env._step_rewards = [0.3, 0.4]
        env._done = True
        # Reset again
        obs = env.reset("easy")
        assert env._step_rewards == []
        assert env._done is False
        assert isinstance(obs, Observation)


# ---------------------------------------------------------------------------
# Unit tests for state()
# ---------------------------------------------------------------------------

class TestState:
    def test_state_returns_observation(self):
        env = WorkspaceEnv()
        env.reset("easy")
        obs = env.state()
        assert isinstance(obs, Observation)

    def test_state_matches_reset_observation(self):
        env = WorkspaceEnv()
        reset_obs = env.reset("easy")
        state_obs = env.state()
        assert reset_obs == state_obs

    def test_state_does_not_mutate_internal_files(self):
        env = WorkspaceEnv()
        env.reset("easy")
        obs1 = env.state()
        # Mutate the returned observation's file list (should not affect env)
        obs1.files.clear()
        obs2 = env.state()
        assert len(obs2.files) > 0

    def test_state_does_not_mutate_internal_folders(self):
        env = WorkspaceEnv()
        env.reset("easy")
        obs1 = env.state()
        # Mutate the returned folders dict
        obs1.folders["root"].clear()
        obs2 = env.state()
        assert len(obs2.folders["root"]) > 0


# ---------------------------------------------------------------------------
# Property 2: Reset produces clean state
# Feature: workspace-organizer, Property 2: Reset produces clean state
# Validates: Requirements 2.5
# ---------------------------------------------------------------------------

@given(st.sampled_from(["easy", "medium", "hard"]))
@settings(max_examples=100)
def test_reset_produces_clean_state(task_name: str) -> None:
    """For any sequence of actions applied to an episode, calling reset(task_name)
    afterward SHALL produce an Observation identical to calling reset(task_name)
    on a fresh environment with no prior actions.

    # Feature: workspace-organizer, Property 2: Reset produces clean state
    Validates: Requirements 2.5
    """
    # Fresh environment baseline
    fresh_env = WorkspaceEnv()
    fresh_obs = fresh_env.reset(task_name)

    # Environment with some simulated prior state mutations
    used_env = WorkspaceEnv()
    used_env.reset(task_name)
    # Simulate state drift: modify internal state directly
    used_env._step_rewards = [0.3, 0.4, 0.0]
    used_env._done = True
    if used_env._files:
        first_id = next(iter(used_env._files))
        used_env._files[first_id] = used_env._files[first_id].model_copy(
            update={"name": "mutated_name.txt"}
        )
    # Now reset — must produce identical observation to fresh env
    reset_obs = used_env.reset(task_name)

    assert reset_obs == fresh_obs
    assert used_env._step_rewards == []
    assert used_env._done is False


# ---------------------------------------------------------------------------
# Property 3: state() is read-only and idempotent
# Feature: workspace-organizer, Property 3: state() is read-only and idempotent
# Validates: Requirements 2.3
# ---------------------------------------------------------------------------

@given(st.sampled_from(["easy", "medium", "hard"]))
@settings(max_examples=100)
def test_state_is_read_only_and_idempotent(task_name: str) -> None:
    """For any environment state, calling state() any number of times SHALL return
    equivalent Observation values and SHALL NOT modify the internal state.

    # Feature: workspace-organizer, Property 3: state() is read-only and idempotent
    Validates: Requirements 2.3
    """
    env = WorkspaceEnv()
    env.reset(task_name)

    # Capture internal state snapshots before calling state()
    files_before = dict(env._files)
    folders_before = {k: list(v) for k, v in env._folders.items()}
    rewards_before = list(env._step_rewards)
    done_before = env._done

    # Call state() multiple times
    obs1 = env.state()
    obs2 = env.state()
    obs3 = env.state()

    # All observations must be equivalent
    assert obs1 == obs2
    assert obs2 == obs3

    # Internal state must be unchanged
    assert env._files == files_before
    assert env._folders == folders_before
    assert env._step_rewards == rewards_before
    assert env._done == done_before


# ===========================================================================
# Task 4 Property Tests
# ===========================================================================

# ---------------------------------------------------------------------------
# Property 1: Invalid reference rejection
# Feature: workspace-organizer, Property 1: Invalid reference rejection
# Validates: Requirements 1.4, 1.5
# ---------------------------------------------------------------------------

@given(st.uuids().map(str))
@settings(max_examples=100)
def test_invalid_reference_rejection(bad_id: str) -> None:
    """For any action referencing a file_id or folder not in current state,
    the environment SHALL return a negative reward and leave the state unchanged.

    # Feature: workspace-organizer, Property 1: Invalid reference rejection
    Validates: Requirements 1.4, 1.5
    """
    for task_name in ["easy", "medium", "hard"]:
        env = WorkspaceEnv()
        env.reset(task_name)

        # Ensure bad_id is not accidentally a real file_id
        if bad_id in env._files:
            continue

        files_before = dict(env._files)
        folders_before = {k: list(v) for k, v in env._folders.items()}

        # rename with bad file_id
        result = env.step(Action(action_type="rename", file_id=bad_id, target="new_name.txt"))
        assert result.reward.score == 0.0  # clamped from -0.2
        assert env._files == files_before
        assert {k: list(v) for k, v in env._folders.items()} == folders_before

        # move with bad file_id
        result = env.step(Action(action_type="move", file_id=bad_id, target="root"))
        assert result.reward.score == 0.0
        assert env._files == files_before
        assert {k: list(v) for k, v in env._folders.items()} == folders_before

        # move with bad folder (valid file_id, bad target folder)
        real_file_id = list(env._files.keys())[0]
        bad_folder = bad_id  # UUID string unlikely to be a folder name
        if bad_folder not in env._folders:
            result = env.step(Action(action_type="move", file_id=real_file_id, target=bad_folder))
            assert result.reward.score == 0.0
            assert env._files == files_before
            assert {k: list(v) for k, v in env._folders.items()} == folders_before

        # delete with bad file_id
        result = env.step(Action(action_type="delete", file_id=bad_id))
        assert result.reward.score == 0.0
        assert env._files == files_before
        assert {k: list(v) for k, v in env._folders.items()} == folders_before

        break  # one task is sufficient per hypothesis example


# ---------------------------------------------------------------------------
# Property 4: step() always returns a StepResult
# Feature: workspace-organizer, Property 4: step() always returns a StepResult
# Validates: Requirements 2.2
# ---------------------------------------------------------------------------

@given(
    st.builds(
        Action,
        action_type=st.text(),
        file_id=st.one_of(st.none(), st.text()),
        target=st.one_of(st.none(), st.text()),
    )
)
@settings(max_examples=100)
def test_step_always_returns_step_result(action: Action) -> None:
    """For any Action value (valid or invalid), step() SHALL return a StepResult.

    # Feature: workspace-organizer, Property 4: step() always returns a StepResult
    Validates: Requirements 2.2
    """
    env = WorkspaceEnv()
    env.reset("easy")
    result = env.step(action)
    assert isinstance(result, StepResult)
    assert hasattr(result, "observation")
    assert hasattr(result, "reward")
    assert hasattr(result, "done")
    assert hasattr(result, "info")


# ---------------------------------------------------------------------------
# Property 5: Rename updates file name
# Feature: workspace-organizer, Property 5: Rename updates file name
# Validates: Requirements 3.1
# ---------------------------------------------------------------------------

@given(st.text(min_size=1))
@settings(max_examples=100)
def test_rename_updates_file_name(target: str) -> None:
    """For any valid file_id and non-empty target, after rename, file.name == target.

    # Feature: workspace-organizer, Property 5: Rename updates file name
    Validates: Requirements 3.1
    """
    env = WorkspaceEnv()
    env.reset("easy")
    file_id = list(env._files.keys())[0]
    result = env.step(Action(action_type="rename", file_id=file_id, target=target))
    assert isinstance(result, StepResult)
    assert env._files[file_id].name == target


# ---------------------------------------------------------------------------
# Property 6: Empty or absent target is always invalid
# Feature: workspace-organizer, Property 6: Empty or absent target is always invalid
# Validates: Requirements 3.3, 4.3
# ---------------------------------------------------------------------------

@given(st.sampled_from(["rename", "create_folder"]))
@settings(max_examples=100)
def test_empty_or_absent_target_is_invalid(action_type: str) -> None:
    """For rename or create_folder with empty/None target, reward.score == 0.0 (clamped from -0.2).

    # Feature: workspace-organizer, Property 6: Empty or absent target is always invalid
    Validates: Requirements 3.3, 4.3
    """
    env = WorkspaceEnv()
    env.reset("easy")
    file_id = list(env._files.keys())[0]
    files_before = dict(env._files)
    folders_before = {k: list(v) for k, v in env._folders.items()}

    # Empty string target
    result = env.step(Action(action_type=action_type, file_id=file_id, target=""))
    assert result.reward.score == 0.0
    assert env._files == files_before
    assert {k: list(v) for k, v in env._folders.items()} == folders_before

    # None target
    result = env.step(Action(action_type=action_type, file_id=file_id, target=None))
    assert result.reward.score == 0.0
    assert env._files == files_before
    assert {k: list(v) for k, v in env._folders.items()} == folders_before


# ---------------------------------------------------------------------------
# Property 7: create_folder adds a new empty folder
# Feature: workspace-organizer, Property 7: create_folder adds a new empty folder
# Validates: Requirements 4.1
# ---------------------------------------------------------------------------

@given(st.text(min_size=1))
@settings(max_examples=100)
def test_create_folder_adds_new_empty_folder(folder_name: str) -> None:
    """For any non-empty folder name not already in state, create_folder adds it with empty list.

    # Feature: workspace-organizer, Property 7: create_folder adds a new empty folder
    Validates: Requirements 4.1
    """
    env = WorkspaceEnv()
    env.reset("easy")

    if folder_name in env._folders:
        return  # skip if name already exists (tested by Property 8)

    result = env.step(Action(action_type="create_folder", target=folder_name))
    assert isinstance(result, StepResult)
    assert folder_name in env._folders
    assert env._folders[folder_name] == []


# ---------------------------------------------------------------------------
# Property 8: Duplicate folder creation is rejected
# Feature: workspace-organizer, Property 8: Duplicate folder creation is rejected
# Validates: Requirements 4.2
# ---------------------------------------------------------------------------

@given(st.sampled_from(["easy", "medium", "hard"]))
@settings(max_examples=100)
def test_duplicate_folder_creation_rejected(task_name: str) -> None:
    """For any existing folder name, create_folder returns reward.score == 0.0 (clamped from -0.2).

    # Feature: workspace-organizer, Property 8: Duplicate folder creation is rejected
    Validates: Requirements 4.2
    """
    env = WorkspaceEnv()
    env.reset(task_name)
    existing_folder = list(env._folders.keys())[0]
    folders_before = {k: list(v) for k, v in env._folders.items()}

    result = env.step(Action(action_type="create_folder", target=existing_folder))
    assert result.reward.score == 0.0
    assert {k: list(v) for k, v in env._folders.items()} == folders_before


# ---------------------------------------------------------------------------
# Property 9: Move updates folder membership
# Feature: workspace-organizer, Property 9: Move updates folder membership
# Validates: Requirements 5.1
# ---------------------------------------------------------------------------

@given(st.sampled_from(["easy", "medium", "hard"]))
@settings(max_examples=100)
def test_move_updates_folder_membership(task_name: str) -> None:
    """For valid file_id in folder A and valid target folder B (A≠B), file_id ends up in B not A.

    # Feature: workspace-organizer, Property 9: Move updates folder membership
    Validates: Requirements 5.1
    """
    env = WorkspaceEnv()
    env.reset(task_name)

    # Create a second folder to move into
    env.step(Action(action_type="create_folder", target="__test_folder__"))

    # Pick a file from root
    file_id = list(env._files.keys())[0]
    # Ensure file is in root
    if file_id not in env._folders.get("root", []):
        return

    result = env.step(Action(action_type="move", file_id=file_id, target="__test_folder__"))
    assert isinstance(result, StepResult)
    assert file_id in env._folders["__test_folder__"]
    assert file_id not in env._folders.get("root", [])


# ---------------------------------------------------------------------------
# Property 10: Move to same folder is rejected
# Feature: workspace-organizer, Property 10: Move to same folder is rejected
# Validates: Requirements 5.3
# ---------------------------------------------------------------------------

@given(st.sampled_from(["easy", "medium", "hard"]))
@settings(max_examples=100)
def test_move_to_same_folder_rejected(task_name: str) -> None:
    """For file_id already in folder F, move to F returns reward.score == 0.0 (clamped from -0.2).

    # Feature: workspace-organizer, Property 10: Move to same folder is rejected
    Validates: Requirements 5.3
    """
    env = WorkspaceEnv()
    env.reset(task_name)

    # All files start in root
    file_id = list(env._files.keys())[0]
    folders_before = {k: list(v) for k, v in env._folders.items()}

    result = env.step(Action(action_type="move", file_id=file_id, target="root"))
    assert result.reward.score == 0.0
    assert {k: list(v) for k, v in env._folders.items()} == folders_before


# ---------------------------------------------------------------------------
# Property 11: Delete removes file from all state
# Feature: workspace-organizer, Property 11: Delete removes file from all state
# Validates: Requirements 6.1
# ---------------------------------------------------------------------------

@given(st.sampled_from(["easy", "medium", "hard"]))
@settings(max_examples=100)
def test_delete_removes_file_from_all_state(task_name: str) -> None:
    """For valid file_id, after delete, file_id not in any folder and not in _files.

    # Feature: workspace-organizer, Property 11: Delete removes file from all state
    Validates: Requirements 6.1
    """
    env = WorkspaceEnv()
    env.reset(task_name)
    file_id = list(env._files.keys())[0]

    result = env.step(Action(action_type="delete", file_id=file_id))
    assert isinstance(result, StepResult)
    assert file_id not in env._files
    for folder_files in env._folders.values():
        assert file_id not in folder_files


# ---------------------------------------------------------------------------
# Property 15: Invalid action type returns penalty without termination
# Feature: workspace-organizer, Property 15: Invalid action type returns penalty without termination
# Validates: Requirements 7.5
# ---------------------------------------------------------------------------

@given(
    st.text().filter(lambda s: s not in ("rename", "create_folder", "move", "delete"))
)
@settings(max_examples=100)
def test_invalid_action_type_returns_penalty_without_termination(action_type: str) -> None:
    """For unrecognized action_type, reward.score == 0.0 (clamped from -0.2) and done=False.

    # Feature: workspace-organizer, Property 15: Invalid action type returns penalty without termination
    Validates: Requirements 7.5
    """
    env = WorkspaceEnv()
    env.reset("easy")
    result = env.step(Action(action_type=action_type))
    assert result.reward.score == 0.0
    assert result.done is False


# ===========================================================================
# Task 5 Property and Unit Tests — Reward Computation
# ===========================================================================

# ---------------------------------------------------------------------------
# Property 13: Per-step reward is bounded
# Feature: workspace-organizer, Property 13: Per-step reward is bounded
# Validates: Requirements 7.2
# ---------------------------------------------------------------------------

@given(
    st.builds(
        Action,
        action_type=st.text(),
        file_id=st.one_of(st.none(), st.text()),
        target=st.one_of(st.none(), st.text()),
    )
)
@settings(max_examples=100)
def test_per_step_reward_is_bounded(action: Action) -> None:
    """For any action, reward.score returned by step() SHALL be in [0.0, 1.0].

    # Feature: workspace-organizer, Property 13: Per-step reward is bounded
    Validates: Requirements 7.2
    """
    env = WorkspaceEnv()
    env.reset("easy")
    result = env.step(action)
    assert 0.0 <= result.reward.score <= 1.0


# ---------------------------------------------------------------------------
# Property 14: Episode score is bounded and deterministic
# Feature: workspace-organizer, Property 14: Episode score is bounded and deterministic
# Validates: Requirements 7.3, 7.4
# ---------------------------------------------------------------------------

@given(st.sampled_from(["easy", "medium", "hard"]))
@settings(max_examples=100)
def test_episode_score_is_bounded_and_deterministic(task_name: str) -> None:
    """For any sequence of actions on any task, episode_score() SHALL be in [0.0, 1.0],
    and running the same action sequence twice SHALL produce identical results.

    # Feature: workspace-organizer, Property 14: Episode score is bounded and deterministic
    Validates: Requirements 7.3, 7.4
    """
    # Fixed action sequence used for both runs
    fixed_actions = [
        Action(action_type="rename", file_id="e001", target="photo_001.jpg"),
        Action(action_type="create_folder", target="test_folder"),
        Action(action_type="move", file_id="e002", target="root"),  # same folder → penalty
        Action(action_type="delete", file_id="e003"),
    ]

    # First run
    env1 = WorkspaceEnv()
    env1.reset(task_name)
    rewards1 = []
    for action in fixed_actions:
        result = env1.step(action)
        rewards1.append(result.reward.score)
    score1 = env1.episode_score()

    # Second run — identical sequence
    env2 = WorkspaceEnv()
    env2.reset(task_name)
    rewards2 = []
    for action in fixed_actions:
        result = env2.step(action)
        rewards2.append(result.reward.score)
    score2 = env2.episode_score()

    assert 0.0 <= score1 <= 1.0
    assert 0.0 <= score2 <= 1.0
    assert rewards1 == rewards2
    assert score1 == score2


# ---------------------------------------------------------------------------
# Property 12: Unexpected deletion is penalized
# Feature: workspace-organizer, Property 12: Unexpected deletion is penalized
# Validates: Requirements 6.3
# ---------------------------------------------------------------------------

@given(st.sampled_from(["e001", "e002", "e003", "e004", "e005"]))
@settings(max_examples=100)
def test_unexpected_deletion_is_penalized(file_id: str) -> None:
    """For any file_id NOT in solution.expected_deletions, delete action SHALL return
    reward.score == 0.0 (clamped from -0.5).

    # Feature: workspace-organizer, Property 12: Unexpected deletion is penalized
    Validates: Requirements 6.3
    """
    env = WorkspaceEnv()
    env.reset("easy")  # easy task has no expected deletions
    assert file_id not in env._solution.expected_deletions

    result = env.step(Action(action_type="delete", file_id=file_id))
    # -0.5 clamped to 0.0
    assert result.reward.score == 0.0


# ---------------------------------------------------------------------------
# Unit tests for reward values on solution-matching actions (Task 5.4)
# ---------------------------------------------------------------------------

class TestSolutionMatchingRewards:
    def test_rename_matching_solution_yields_0_3(self):
        """Rename matching solution expected name yields reward.score == 0.3."""
        env = WorkspaceEnv()
        env.reset("easy")
        # e001 solution rename: "photo_001.jpg"
        result = env.step(Action(action_type="rename", file_id="e001", target="photo_001.jpg"))
        assert result.reward.score == pytest.approx(0.3)

    def test_move_matching_solution_yields_0_4(self):
        """Move matching solution placement yields reward.score == 0.4."""
        env = WorkspaceEnv()
        env.reset("medium")
        # Create the target folder first
        env.step(Action(action_type="create_folder", target="tax_documents"))
        # m001 solution placement: "tax_documents"
        result = env.step(Action(action_type="move", file_id="m001", target="tax_documents"))
        assert result.reward.score == pytest.approx(0.4)

    def test_delete_matching_expected_duplicate_yields_0_5(self):
        """Delete matching expected deletion yields reward.score == 0.5."""
        env = WorkspaceEnv()
        env.reset("hard")
        # h008 is in expected_deletions
        result = env.step(Action(action_type="delete", file_id="h008"))
        assert result.reward.score == pytest.approx(0.5)
