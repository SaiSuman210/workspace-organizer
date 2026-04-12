"""FastAPI server exposing the WorkspaceEnv over HTTP for OpenEnv compatibility."""
from __future__ import annotations

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from env.env import WorkspaceEnv
from env.models import Action, Observation, StepResult
from env.tasks import TASKS

app = FastAPI(title="Workspace Organizer", version="1.0.0")

# One shared env instance per server process
_env = WorkspaceEnv()


class ResetRequest(BaseModel):
    task: str = "easy"


class StepRequest(BaseModel):
    action: Action


@app.post("/reset", response_model=Observation)
def reset(req: ResetRequest = ResetRequest()):
    """Reset the environment and return the initial observation."""
    try:
        obs = _env.reset(req.task)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return obs


@app.post("/step", response_model=StepResult)
def step(req: StepRequest):
    """Apply an action and return the step result."""
    return _env.step(req.action)


@app.get("/state", response_model=Observation)
def state():
    """Return the current observation without modifying state."""
    return _env.state()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tasks")
def list_tasks():
    """List all available tasks."""
    return {"tasks": list(TASKS.keys())}


def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
