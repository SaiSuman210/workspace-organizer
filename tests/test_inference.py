"""Tests for inference.py — log format, error handling, and integration.

Covers:
- Property 16: Log lines are valid single-line structured text
- Unit tests for [START] / [END] log format
- Malformed agent JSON produces synthetic invalid action (no crash)
- Integration test: full episode with scripted agent
"""
from __future__ import annotations

# Feature: workspace-organizer, Property 16: Log lines are valid single-line JSON

import contextlib
import io
import json
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from env.models import Action
from inference import format_prompt, parse_action, run_episode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_client(action_json: str) -> MagicMock:
    """Return a mock OpenAI client whose completions always return action_json."""
    mock_choice = MagicMock()
    mock_choice.message.content = action_json
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_completion
    return mock_client


def _capture_lines(task_name: str, mock_client: MagicMock) -> list[str]:
    """Run run_episode with a mocked OpenAI client; return non-empty stdout lines."""
    buf = io.StringIO()
    with patch("inference.OpenAI", return_value=mock_client):
        with contextlib.redirect_stdout(buf):
            run_episode(task_name)
    return [line for line in buf.getvalue().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Property 16: Log lines are valid single-line structured text
# Feature: workspace-organizer, Property 16: Log lines are valid single-line JSON
# Validates: Requirements 10.2, 10.6
# ---------------------------------------------------------------------------

@given(st.sampled_from(["easy", "medium", "hard"]))
@settings(max_examples=10)
def test_log_lines_are_valid_single_line(task_name: str) -> None:
    """Every line emitted to stdout SHALL be a single line with no embedded newlines.

    # Feature: workspace-organizer, Property 16: Log lines are valid single-line JSON
    Validates: Requirements 10.2, 10.6
    """
    action_json = json.dumps({"action_type": "create_folder", "target": "test_folder"})
    mock_client = _make_mock_client(action_json)
    lines = _capture_lines(task_name, mock_client)

    assert len(lines) >= 2, "Expected at least [START] and [END] lines"
    for line in lines:
        assert "\n" not in line
        assert "\r" not in line
        assert line.startswith(("[START]", "[STEP]", "[END]")), f"Unexpected line: {line!r}"


# ---------------------------------------------------------------------------
# Unit tests for log format (Task 7.2)
# ---------------------------------------------------------------------------

class TestLogFormat:
    def test_start_log_contains_task(self):
        """[START] line must contain task= and env= fields."""
        mock_client = _make_mock_client(json.dumps({"action_type": "create_folder", "target": "docs"}))
        lines = _capture_lines("easy", mock_client)
        start = next(l for l in lines if l.startswith("[START]"))
        assert "task=easy" in start
        assert "env=workspace-organizer" in start

    def test_end_log_contains_score(self):
        """[END] line must contain score= field."""
        mock_client = _make_mock_client(json.dumps({"action_type": "create_folder", "target": "docs"}))
        lines = _capture_lines("easy", mock_client)
        end = next(l for l in lines if l.startswith("[END]"))
        assert "score=" in end

    def test_step_log_contains_required_fields(self):
        """[STEP] lines must contain step=, action=, reward=, done= fields."""
        mock_client = _make_mock_client(
            json.dumps({"action_type": "rename", "file_id": "e001", "target": "photo_001.jpg"})
        )
        lines = _capture_lines("easy", mock_client)
        step_lines = [l for l in lines if l.startswith("[STEP]")]
        assert len(step_lines) >= 1
        for line in step_lines:
            assert "step=" in line
            assert "action=" in line
            assert "reward=" in line
            assert "done=" in line

    def test_malformed_agent_json_no_crash(self):
        """Malformed agent JSON must produce a synthetic invalid action, not a crash."""
        mock_client = _make_mock_client("this is not json at all }{")
        lines = _capture_lines("easy", mock_client)
        events = [l.split("]")[0][1:] for l in lines]
        assert "START" in events
        assert "END" in events
        step_lines = [l for l in lines if l.startswith("[STEP]")]
        assert len(step_lines) >= 1
        # parse_error action should appear in the action JSON
        assert "parse_error" in step_lines[0]

    def test_empty_agent_response_no_crash(self):
        """Empty agent response must not crash."""
        mock_client = _make_mock_client("")
        lines = _capture_lines("easy", mock_client)
        events = [l.split("]")[0][1:] for l in lines]
        assert "START" in events
        assert "END" in events

    def test_parse_action_valid_json(self):
        """parse_action returns correct Action for valid JSON."""
        action, err = parse_action('{"action_type": "rename", "file_id": "e001", "target": "photo.jpg"}')
        assert action.action_type == "rename"
        assert action.file_id == "e001"
        assert action.target == "photo.jpg"
        assert err is None

    def test_parse_action_invalid_json_returns_parse_error(self):
        """parse_action returns parse_error Action for invalid JSON."""
        action, err = parse_action("not valid json")
        assert action.action_type == "parse_error"
        assert err is not None

    def test_parse_action_missing_action_type_returns_parse_error(self):
        """parse_action returns parse_error Action when action_type is missing."""
        action, err = parse_action('{"file_id": "e001"}')
        assert action.action_type == "parse_error"


# ---------------------------------------------------------------------------
# Integration tests (Task 7.3)
# ---------------------------------------------------------------------------

class TestFullEpisodeIntegration:
    def _make_scripted_client(self, actions: list[dict]) -> MagicMock:
        def make_completion(action_dict):
            mock_choice = MagicMock()
            mock_choice.message.content = json.dumps(action_dict)
            mock_completion = MagicMock()
            mock_completion.choices = [mock_choice]
            return mock_completion

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            make_completion(a) for a in actions
        ] + [make_completion({"action_type": "noop"}) for _ in range(50)]
        return mock_client

    def _get_score(self, lines: list[str]) -> float:
        end = next(l for l in lines if l.startswith("[END]"))
        # parse score=0.300 from the line
        for part in end.split():
            if part.startswith("score="):
                return float(part.split("=")[1])
        raise ValueError(f"No score= in: {end}")

    def test_full_episode_easy_scripted_agent(self):
        scripted = [
            {"action_type": "rename", "file_id": "e001", "target": "photo_001.jpg"},
            {"action_type": "rename", "file_id": "e002", "target": "photo_002.jpg"},
            {"action_type": "rename", "file_id": "e003", "target": "invoice_scan.pdf"},
            {"action_type": "rename", "file_id": "e004", "target": "meeting_notes.txt"},
            {"action_type": "rename", "file_id": "e005", "target": "family_portrait.png"},
        ]
        lines = _capture_lines("easy", self._make_scripted_client(scripted))
        score = self._get_score(lines)
        assert 0.0 <= score <= 1.0

    def test_full_episode_hard_scripted_agent(self):
        scripted = [
            {"action_type": "create_folder", "target": "projects"},
            {"action_type": "create_folder", "target": "finance"},
            {"action_type": "create_folder", "target": "photos"},
            {"action_type": "create_folder", "target": "contracts"},
            {"action_type": "create_folder", "target": "design"},
            {"action_type": "create_folder", "target": "notes"},
            {"action_type": "create_folder", "target": "hr"},
            {"action_type": "move", "file_id": "h001", "target": "projects"},
            {"action_type": "move", "file_id": "h002", "target": "finance"},
            {"action_type": "move", "file_id": "h003", "target": "photos"},
            {"action_type": "move", "file_id": "h004", "target": "contracts"},
            {"action_type": "move", "file_id": "h005", "target": "projects"},
            {"action_type": "move", "file_id": "h006", "target": "design"},
            {"action_type": "move", "file_id": "h007", "target": "notes"},
            {"action_type": "move", "file_id": "h009", "target": "finance"},
            {"action_type": "move", "file_id": "h011", "target": "projects"},
            {"action_type": "move", "file_id": "h012", "target": "hr"},
            {"action_type": "delete", "file_id": "h008"},
            {"action_type": "delete", "file_id": "h010"},
        ]
        lines = _capture_lines("hard", self._make_scripted_client(scripted))
        score = self._get_score(lines)
        assert 0.0 <= score <= 1.0
        assert score > 0.0

    def test_episode_order_start_steps_end(self):
        mock_client = _make_mock_client(json.dumps({"action_type": "create_folder", "target": "misc"}))
        lines = _capture_lines("easy", mock_client)
        assert lines[0].startswith("[START]")
        assert lines[-1].startswith("[END]")
        assert any(l.startswith("[STEP]") for l in lines)

    def test_episode_score_in_range_all_tasks(self):
        action_json = json.dumps({"action_type": "create_folder", "target": "misc"})
        for task_name in ["easy", "medium", "hard"]:
            lines = _capture_lines(task_name, _make_mock_client(action_json))
            end = next(l for l in lines if l.startswith("[END]"))
            for part in end.split():
                if part.startswith("score="):
                    score = float(part.split("=")[1])
                    assert 0.0 <= score <= 1.0
