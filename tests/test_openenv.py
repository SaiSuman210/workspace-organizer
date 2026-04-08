"""
Unit tests for openenv.yaml validity.
Requirements: 11.1, 11.2
"""
import yaml
import pytest
from pathlib import Path


OPENENV_YAML_PATH = Path(__file__).parent.parent / "openenv.yaml"


def load_openenv_yaml():
    with open(OPENENV_YAML_PATH, "r") as f:
        return yaml.safe_load(f)


def test_openenv_yaml_parses_without_errors():
    """openenv.yaml must parse as valid YAML with no syntax errors."""
    data = load_openenv_yaml()
    assert data is not None


def test_openenv_yaml_required_fields_present():
    """openenv.yaml must contain all required top-level fields."""
    data = load_openenv_yaml()
    required_fields = {"name", "version", "description", "tasks", "reward_range"}
    for field in required_fields:
        assert field in data, f"Missing required field: '{field}'"


def test_openenv_yaml_tasks_is_list_with_all_difficulties():
    """tasks must be a list containing 'easy', 'medium', and 'hard'."""
    data = load_openenv_yaml()
    tasks = data["tasks"]
    assert isinstance(tasks, list), "tasks must be a list"
    assert "easy" in tasks
    assert "medium" in tasks
    assert "hard" in tasks


def test_openenv_yaml_reward_range_has_min_and_max():
    """reward_range must have both 'min' and 'max' keys."""
    data = load_openenv_yaml()
    reward_range = data["reward_range"]
    assert isinstance(reward_range, dict), "reward_range must be a mapping"
    assert "min" in reward_range, "reward_range missing 'min'"
    assert "max" in reward_range, "reward_range missing 'max'"


def test_openenv_yaml_reward_range_values():
    """reward_range min should be 0.0 and max should be 1.0."""
    data = load_openenv_yaml()
    reward_range = data["reward_range"]
    assert reward_range["min"] == 0.0
    assert reward_range["max"] == 1.0
