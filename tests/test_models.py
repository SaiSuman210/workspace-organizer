"""Unit tests for Pydantic schema validation in env/models.py.

Requirements: 1.1, 2.4
"""
import pytest
from pydantic import ValidationError

from env.models import (
    Action,
    File,
    Observation,
    Reward,
    StepResult,
    TaskSolution,
)


# ---------------------------------------------------------------------------
# File
# ---------------------------------------------------------------------------

class TestFile:
    def test_valid_file(self):
        f = File(
            id="f001",
            name="photo.jpg",
            type="image",
            date="2024-03-15",
            summary="beach vacation photo",
            size=2048,
        )
        assert f.id == "f001"
        assert f.name == "photo.jpg"
        assert f.size == 2048

    def test_file_serializes_to_dict(self):
        f = File(id="f002", name="doc.pdf", type="document",
                 date="2024-01-01", summary="tax return", size=512)
        d = f.model_dump()
        assert d["id"] == "f002"
        assert d["size"] == 512

    def test_file_round_trips_json(self):
        f = File(id="f003", name="archive.zip", type="archive",
                 date="2023-12-31", summary="backup", size=10240)
        restored = File.model_validate_json(f.model_dump_json())
        assert restored == f

    def test_file_missing_id_raises(self):
        with pytest.raises(ValidationError):
            File(name="x.jpg", type="image", date="2024-01-01",
                 summary="s", size=1)

    def test_file_missing_name_raises(self):
        with pytest.raises(ValidationError):
            File(id="f1", type="image", date="2024-01-01", summary="s", size=1)

    def test_file_missing_size_raises(self):
        with pytest.raises(ValidationError):
            File(id="f1", name="x.jpg", type="image",
                 date="2024-01-01", summary="s")

    def test_file_size_must_be_int(self):
        with pytest.raises(ValidationError):
            File(id="f1", name="x.jpg", type="image",
                 date="2024-01-01", summary="s", size="big")


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class TestAction:
    def test_valid_rename_action(self):
        a = Action(action_type="rename", file_id="f001", target="new_name.jpg")
        assert a.action_type == "rename"
        assert a.file_id == "f001"
        assert a.target == "new_name.jpg"

    def test_valid_create_folder_action(self):
        a = Action(action_type="create_folder", target="Photos")
        assert a.file_id is None

    def test_valid_delete_action(self):
        a = Action(action_type="delete", file_id="f002")
        assert a.target is None

    def test_action_optional_fields_default_none(self):
        a = Action(action_type="delete")
        assert a.file_id is None
        assert a.target is None

    def test_action_missing_action_type_raises(self):
        with pytest.raises(ValidationError):
            Action(file_id="f001", target="folder")

    def test_action_round_trips_json(self):
        a = Action(action_type="move", file_id="f005", target="Documents")
        restored = Action.model_validate_json(a.model_dump_json())
        assert restored == a


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class TestObservation:
    def _make_file(self, fid: str) -> File:
        return File(id=fid, name=f"{fid}.txt", type="document",
                    date="2024-06-01", summary="test file", size=100)

    def test_valid_observation(self):
        obs = Observation(
            files=[self._make_file("f1"), self._make_file("f2")],
            folders={"root": ["f1", "f2"]},
            instruction="Organize files",
        )
        assert len(obs.files) == 2
        assert "root" in obs.folders

    def test_observation_empty_files(self):
        obs = Observation(files=[], folders={"root": []}, instruction="Nothing to do")
        assert obs.files == []

    def test_observation_missing_instruction_raises(self):
        with pytest.raises(ValidationError):
            Observation(files=[], folders={})

    def test_observation_missing_folders_raises(self):
        with pytest.raises(ValidationError):
            Observation(files=[], instruction="hi")

    def test_observation_round_trips_json(self):
        obs = Observation(
            files=[self._make_file("f1")],
            folders={"root": ["f1"]},
            instruction="Test",
        )
        restored = Observation.model_validate_json(obs.model_dump_json())
        assert restored == obs


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class TestReward:
    def test_valid_reward_with_message(self):
        r = Reward(score=0.4, message="Good move")
        assert r.score == 0.4
        assert r.message == "Good move"

    def test_valid_reward_without_message(self):
        r = Reward(score=0.0)
        assert r.message is None

    def test_reward_missing_score_raises(self):
        with pytest.raises(ValidationError):
            Reward(message="oops")

    def test_reward_round_trips_json(self):
        r = Reward(score=1.0, message="Perfect")
        restored = Reward.model_validate_json(r.model_dump_json())
        assert restored == r


# ---------------------------------------------------------------------------
# StepResult
# ---------------------------------------------------------------------------

class TestStepResult:
    def _make_obs(self) -> Observation:
        return Observation(files=[], folders={"root": []}, instruction="Test")

    def test_valid_step_result(self):
        sr = StepResult(
            observation=self._make_obs(),
            reward=Reward(score=0.3),
            done=False,
            info={"step": 1},
        )
        assert sr.done is False
        assert sr.reward.score == 0.3

    def test_step_result_done_true(self):
        sr = StepResult(
            observation=self._make_obs(),
            reward=Reward(score=1.0),
            done=True,
            info={},
        )
        assert sr.done is True

    def test_step_result_missing_observation_raises(self):
        with pytest.raises(ValidationError):
            StepResult(reward=Reward(score=0.0), done=False, info={})

    def test_step_result_missing_done_raises(self):
        with pytest.raises(ValidationError):
            StepResult(observation=self._make_obs(), reward=Reward(score=0.0), info={})

    def test_step_result_round_trips_json(self):
        sr = StepResult(
            observation=self._make_obs(),
            reward=Reward(score=0.5, message="ok"),
            done=False,
            info={"extra": "data"},
        )
        restored = StepResult.model_validate_json(sr.model_dump_json())
        assert restored == sr


# ---------------------------------------------------------------------------
# TaskSolution
# ---------------------------------------------------------------------------

class TestTaskSolution:
    def test_valid_task_solution(self):
        ts = TaskSolution(
            expected_renames={"f001": "photo_001.jpg"},
            expected_placements={"f002": "Documents"},
            expected_deletions={"f003", "f004"},
        )
        assert "f001" in ts.expected_renames
        assert "f003" in ts.expected_deletions

    def test_task_solution_empty_fields(self):
        ts = TaskSolution(
            expected_renames={},
            expected_placements={},
            expected_deletions=set(),
        )
        assert ts.expected_deletions == set()

    def test_task_solution_missing_expected_renames_raises(self):
        with pytest.raises(ValidationError):
            TaskSolution(expected_placements={}, expected_deletions=set())

    def test_task_solution_missing_expected_placements_raises(self):
        with pytest.raises(ValidationError):
            TaskSolution(expected_renames={}, expected_deletions=set())

    def test_task_solution_missing_expected_deletions_raises(self):
        with pytest.raises(ValidationError):
            TaskSolution(expected_renames={}, expected_placements={})

    def test_task_solution_round_trips_json(self):
        ts = TaskSolution(
            expected_renames={"f1": "clean.jpg"},
            expected_placements={"f2": "Photos"},
            expected_deletions={"f3"},
        )
        restored = TaskSolution.model_validate_json(ts.model_dump_json())
        assert restored == ts
