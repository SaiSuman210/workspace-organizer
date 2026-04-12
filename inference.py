"""Inference runner for the Workspace Organizer benchmark.

Stdout format:
    [START] task=<task_name> env=workspace-organizer model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
"""
from __future__ import annotations

import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from env.env import WorkspaceEnv
from env.models import Action

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = "workspace-organizer"
MAX_STEPS = 50
SUCCESS_SCORE_THRESHOLD = 0.1

SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI agent organizing a simulated file system.
    You will receive the current state of files and folders as JSON.
    Respond with a single JSON object representing the action to take.
    Valid action_types: 'rename', 'move', 'create_folder', 'delete'.
    For 'rename': provide file_id and target (new name).
    For 'move': provide file_id and target (folder name).
    For 'create_folder': provide target (folder name).
    For 'delete': provide file_id.
    Example: {"action_type": "rename", "file_id": "e001", "target": "photo_001.jpg"}
    Respond ONLY with a valid JSON object, no explanation.
""").strip()


def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def get_action(client: OpenAI, messages: list) -> tuple[str, Optional[str]]:
    """Call the LLM and return (action_json_str, error_or_None)."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
        )
        text = (completion.choices[0].message.content or "").strip()
        data = json.loads(text)
        if "action_type" not in data:
            raise ValueError("Missing action_type")
        return text, None
    except Exception as exc:
        return json.dumps({"action_type": "parse_error"}), str(exc)


def run_episode(task_name: str) -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = WorkspaceEnv()
    obs = env.reset(task_name)

    log_start(task=task_name, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    try:
        for step in range(1, MAX_STEPS + 1):
            obs_dict = obs.model_dump()
            prompt = f"Current workspace state:\n{json.dumps(obs_dict, indent=2)}\n\nWhat action should be taken next?"
            messages.append({"role": "user", "content": prompt})

            action_str, error = get_action(client, messages)
            messages.append({"role": "assistant", "content": action_str})

            action_data = json.loads(action_str)
            action = Action(**action_data)
            result = env.step(action)

            reward = result.reward.score
            done = result.done
            obs = result.observation

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        log_step(step=steps_taken + 1, action=json.dumps({"action_type": "error"}),
                 reward=0.0, done=True, error=str(exc))

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all", choices=["easy", "medium", "hard", "all"])
    args = parser.parse_args()
    if args.task == "all":
        for task in ["easy", "medium", "hard"]:
            run_episode(task)
    else:
        run_episode(args.task)
