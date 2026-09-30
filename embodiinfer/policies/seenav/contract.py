"""SeeNav-Agent contracts and the discrete action grammar."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from ...types import Observation
from ..base import PrefixState

SEENAV_CHECKPOINT = "wangzc9865/SeeNav-Agent"
SEENAV_REVISION = "b1024343452d2cf42bd23b4dc6e8efe01313b549"
SEENAV_ACTION_NAMES = (
    "Move forward by 0.25",
    "Move backward by 0.25",
    "Move rightward by 0.25",
    "Move leftward by 0.25",
    "Rotate to the right by 90 degrees",
    "Rotate to the left by 90 degrees",
    "Tilt the camera upward by 30 degrees",
    "Tilt the camera downward by 30 degrees",
)
SEENAV_MAX_ACTION_ID = len(SEENAV_ACTION_NAMES) - 1
SEENAV_MAX_PLAN_ACTIONS = 8
SEENAV_MAX_BATCH_SIZE = 8
SEENAV_MESSAGE_WINDOW = 5


class SeeNavOutputError(ValueError):
    """The model emitted text outside SeeNav's compact JSON action contract."""


@dataclass(frozen=True)
class SeeNavTurn:
    """One completed visual prompt and the model response kept for chat history."""

    views: tuple[torch.Tensor, ...]
    prompt: str
    response: str
    action_ids: tuple[int, ...]


@dataclass(frozen=True)
class SeeNavMemory:
    """Episode-local history for one recurrent SeeNav session."""

    turns: tuple[SeeNavTurn, ...] = ()

    @property
    def seq_len(self) -> int:
        return len(self.turns)

    def to(self, device: torch.device | str) -> SeeNavMemory:
        # Images remain CPU-owned snapshots; the runner converts only the active prompt.
        torch.device(device)
        return self

    def expand(self, num_samples: int) -> SeeNavMemory:
        if type(num_samples) is not int or num_samples != 1:
            raise ValueError("SeeNav only supports B=1 memory expansion")
        return self

    def compact(self, keep: Sequence[int] | torch.Tensor | None = None) -> SeeNavMemory:
        if keep is None:
            return self
        if isinstance(keep, torch.Tensor):
            if keep.ndim != 1:
                raise ValueError("SeeNav compact indices must be rank one")
            indices = tuple(int(value) for value in keep.detach().cpu().tolist())
        else:
            indices = tuple(keep)
        if indices != (0,) or any(type(value) is not int for value in indices):
            raise ValueError("SeeNav only supports compact([0])")
        return self


@dataclass
class SeeNavBatch:
    """The observation/request layout consumed by the SeeNav runner."""

    observations: list[Observation]
    request_ids: list[str]

    def __post_init__(self) -> None:
        if not 1 <= len(self.observations) <= SEENAV_MAX_BATCH_SIZE:
            raise ValueError(f"SeeNav batch size must be between 1 and {SEENAV_MAX_BATCH_SIZE}")
        if len(self.request_ids) != len(self.observations):
            raise ValueError("SeeNav observations and request_ids must have identical lengths")
        if len(set(self.request_ids)) != len(self.request_ids):
            raise ValueError("SeeNav request_ids must be unique within a batch")

    @property
    def batch_size(self) -> int:
        return len(self.observations)

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> SeeNavBatch:
        del device, dtype
        return self


@dataclass
class SeeNavPrefix:
    """A B=1 observation plus its committed episode history."""

    observation: Observation
    memory: SeeNavMemory
    batch_size: int = 1

    def to(self, device: torch.device | str) -> SeeNavPrefix:
        del device
        return self

    def expand(self, num_samples: int) -> PrefixState:
        if num_samples != 1:
            raise NotImplementedError("SeeNav does not support candidate expansion")
        return self


@dataclass(frozen=True)
class SeeNavBatchPrefix:
    """A padded processor batch plus one committed memory per session."""

    observations: tuple[Observation, ...]
    memories: tuple[SeeNavMemory, ...]
    batch_size: int

    def __post_init__(self) -> None:
        if self.batch_size != len(self.observations) or self.batch_size != len(self.memories):
            raise ValueError("SeeNav batch prefix rows and batch_size must agree")
        if not 1 <= self.batch_size <= SEENAV_MAX_BATCH_SIZE:
            raise ValueError(f"SeeNav prefix batch size must be between 1 and {SEENAV_MAX_BATCH_SIZE}")

    def to(self, device: torch.device | str) -> SeeNavBatchPrefix:
        del device
        return self

    def expand(self, num_samples: int) -> PrefixState:
        if num_samples != 1:
            raise NotImplementedError("SeeNav does not support candidate expansion")
        return self


def _strip_code_fence(text: str) -> str:
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    return value


def parse_seenav_actions(text: str, *, max_actions: int = SEENAV_MAX_PLAN_ACTIONS) -> torch.Tensor:
    """Parse compact SeeNav JSON into ``[num_actions, 2]`` discrete action rows.

    The compact contract is ``{"actions": [0, 2]}``. The legacy
    ``executable_plan`` object is still accepted so recorded responses remain
    readable, but new prompts request only the compact form. The first output
    column is the upstream action id and the second is reserved for the
    model-neutral navigation payload.
    """

    if not isinstance(text, str) or not text.strip():
        raise SeeNavOutputError("SeeNav output is empty")
    try:
        payload = json.loads(_strip_code_fence(text))
    except json.JSONDecodeError as exc:
        raise SeeNavOutputError(f"invalid SeeNav JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise SeeNavOutputError("SeeNav output must be a JSON object")
    compact = payload.get("actions")
    if compact is not None:
        if not isinstance(compact, list) or not compact:
            raise SeeNavOutputError("SeeNav actions must be a non-empty list")
        if len(compact) > max_actions:
            raise SeeNavOutputError(f"SeeNav actions has {len(compact)} actions; maximum is {max_actions}")
        action_ids: list[int] = []
        for index, action_id in enumerate(compact):
            if isinstance(action_id, bool) or not isinstance(action_id, int):
                raise SeeNavOutputError(f"actions[{index}] must be an integer")
            if not 0 <= action_id <= SEENAV_MAX_ACTION_ID:
                raise SeeNavOutputError(f"action id {action_id} outside [0, {SEENAV_MAX_ACTION_ID}]")
            action_ids.append(action_id)
        return torch.tensor([[float(action_id), 0.0] for action_id in action_ids], dtype=torch.float32)

    plan = payload.get("executable_plan")
    if not isinstance(plan, list) or not plan:
        raise SeeNavOutputError("SeeNav actions must be a non-empty list")
    if len(plan) > max_actions:
        raise SeeNavOutputError(f"SeeNav executable_plan has {len(plan)} actions; maximum is {max_actions}")
    action_ids: list[int] = []
    for index, item in enumerate(plan):
        if not isinstance(item, dict):
            raise SeeNavOutputError(f"executable_plan[{index}] must be an object")
        action_id = item.get("action_id")
        if isinstance(action_id, bool) or not isinstance(action_id, int):
            raise SeeNavOutputError(f"executable_plan[{index}].action_id must be an integer")
        if not 0 <= action_id <= SEENAV_MAX_ACTION_ID:
            raise SeeNavOutputError(f"action id {action_id} outside [0, {SEENAV_MAX_ACTION_ID}]")
        action_name = item.get("action_name")
        if not isinstance(action_name, str) or not action_name.strip():
            raise SeeNavOutputError(f"executable_plan[{index}].action_name is required")
        action_ids.append(action_id)
    return torch.tensor([[float(action_id), 0.0] for action_id in action_ids], dtype=torch.float32)


__all__ = [
    "SEENAV_ACTION_NAMES",
    "SEENAV_CHECKPOINT",
    "SEENAV_MAX_PLAN_ACTIONS",
    "SEENAV_MAX_BATCH_SIZE",
    "SEENAV_MESSAGE_WINDOW",
    "SEENAV_REVISION",
    "SeeNavBatch",
    "SeeNavBatchPrefix",
    "SeeNavMemory",
    "SeeNavOutputError",
    "SeeNavPrefix",
    "SeeNavTurn",
    "parse_seenav_actions",
]
