"""Inference runner for the Workspace Organizer benchmark.

Drives a single episode end-to-end:
1. Parses --task CLI arg (default "easy")
2. Calls env.reset() and emits [START] log line to stdout
3. Loop: format observation as prompt → call HF router → parse Action → step env → emit [STEP] log
4. On done=True or step budget exhausted → emit [END] log with final episode score

Stdout format:
    [START] task=<task_name> env=workspace-organizer model=<model_name>
    [STEP]  step=<n> action=<action_json> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional

from openai import OpenAI

from env.env import WorkspaceEnv
from env.models import Action

MAX_STEPS = 50
BENCHMARK = "workspace-organizer"

SYSTEM_PROMPT = (
    "You are an AI agent organizing a simulated file system. "
    "You will receive the current state of files and folders as JSON. "
    "Respond with a single JSON object representing the action to take. "
    "Valid action_types: 'rename', 'move', 'create_folder', 'delete'. "
    "For 'rename': provide file_id and target (new name). "
    "For 'move': provide file_id and target (folder name). "
    "For 'create_folder': provide target (folder name). "
    "For 'delete': provide file_id. "
    'Example: {"action_type": "rename", "file_id": "e001", "target": "photo_001.jpg"}'
)


def format_prompt(observation) -> str:
    """Format observation as a user message prompt for the agent."""
    obs_dict = observation.model_dump() if hasattr(observation, "model_dump") else observation
    return (
        f"Current workspace state:\n{json.dumps(obs_dict, indent=2)}\n\n"
        "What action should be taken next? Respond with a single JSON object."
    )


def parse_action(response_text: str) -> tuple[Action, Optional[str]]:
    """Try to parse JSON response into Action; fall back to parse_error action.

    Returns (action, error_message_or_None).
    """
    try:
        data = json.loads(response_text.strip())
        return Action(**data), None
    except Exception as exc:
        return Action(action_type="parse_error"), str(exc)


def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)


def log_step(step: int, action: Action, reward: float, done: bool, error: Optional[str]) -> None:
    action_str = json.dumps(action.model_dump())
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def run_episode(task_name: str) -> None:
    api_base_url = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
    api_key = os.environ.get("HF_TOKEN")
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

    env = WorkspaceEnv()
    obs = env.reset(task_name)

    log_start(task=task_name, model=model_name)

    client = OpenAI(base_url=api_base_url, api_key=api_key)

    rewards: List[float] = []
    steps_taken = 0
    # Keep message history so the model doesn't repeat actions
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for step in range(1, MAX_STEPS + 1):
        prompt = format_prompt(obs)
        messages.append({"role": "user", "content": prompt})
        parse_error: Optional[str] = None

        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
            )
            response_text = (completion.choices[0].message.content or "").strip()
            # Add assistant reply to history
            messages.append({"role": "assistant", "content": response_text})
        except Exception as exc:
            response_text = ""
            parse_error = str(exc)
            messages.append({"role": "assistant", "content": ""})

        action, action_error = parse_action(response_text)
        error = parse_error or action_error

        result = env.step(action)
        obs = result.observation
        rewards.append(result.reward.score)
        steps_taken = step

        log_step(step=step, action=action, reward=result.reward.score, done=result.done, error=error)

        if result.done:
            break

    score = env.episode_score()
    success = score > 0.0
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a Workspace Organizer episode.")
    parser.add_argument("--task", default="easy", choices=["easy", "medium", "hard"])
    args = parser.parse_args()
    run_episode(args.task)
