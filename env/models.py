from __future__ import annotations

from typing import Dict, List, Optional, Set

from pydantic import BaseModel


class File(BaseModel):
    id: str
    name: str
    type: str        # e.g. "image", "document", "archive"
    date: str        # ISO 8601 date string
    summary: str     # short description used for duplicate detection
    size: int        # bytes, used for duplicate detection


class Action(BaseModel):
    action_type: str   # "rename" | "move" | "create_folder" | "delete"
    file_id: Optional[str] = None
    target: Optional[str] = None


class Observation(BaseModel):
    files: List[File]
    folders: Dict[str, List[str]]   # folder_name -> [file_id, ...]
    instruction: str


class Reward(BaseModel):
    score: float
    message: Optional[str] = None


class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict


class TaskSolution(BaseModel):
    expected_renames: Dict[str, str]      # file_id -> expected name
    expected_placements: Dict[str, str]   # file_id -> expected folder
    expected_deletions: Set[str]          # file_ids to delete
