"""SeeNav-Agent policy, decoder, and factory registration."""

from __future__ import annotations

from pathlib import Path

from ...types import DecodeTrace
from ..base import MemoryState, VLAPolicy
from ..config import VLAPolicyConfig
from ..decoder import AutoregressiveDecoder, DecodeResult
from ..factory import register_policy
from .contract import (
    SEENAV_MAX_PLAN_ACTIONS,
    SeeNavBatch,
    SeeNavMemory,
    SeeNavPrefix,
    SeeNavTurn,
    parse_seenav_actions,
)
from .runner import SeeNavRunner


class SeeNavDecoder(AutoregressiveDecoder):
    """Eager JSON generation and action parsing for one recurrent episode."""

    def __init__(self, policy: SeeNavPolicy):
        self.policy = policy

    def decode(
        self,
        state,
        prefix,
        num_steps,
        bucket,
        graphs,
        *,
        generator=None,
        cancelled=None,
    ):
        del state, num_steps, bucket, graphs, generator
        if cancelled is not None and cancelled():
            from ...exceptions import SessionCancelledError

            raise SessionCancelledError("navigation request cancelled")
        prompt = self.policy.runner.build_prompt(prefix.observation, prefix.memory)
        text, token_ids, entropies = self.policy.runner.infer(prefix.observation, prefix.memory)
        actions = parse_seenav_actions(text)
        views = tuple(view.detach().cpu().clone() for view in prefix.observation.images[:2])
        turns = (*prefix.memory.turns, SeeNavTurn(views, prompt, text, tuple(int(row[0]) for row in actions)))
        memory = SeeNavMemory(turns=turns[-4:])
        trace = DecodeTrace(
            token_ids=token_ids.detach().long().cpu(),
            text=text,
            parsed_actions=actions.tolist(),
            stop_reason="model",
            meta={
                "profile": "seenav",
                "runtime_mode": "eager",
                "cuda_graph_requested": self.policy.cuda_graph_requested,
                "cuda_graph_confirmed": self.policy.cuda_graph_enabled,
                "token_entropies": entropies,
            },
        )
        return DecodeResult(actions=actions.unsqueeze(0), next_memory=memory, traces=[trace])


class SeeNavPolicy(VLAPolicy):
    """B=1 recurrent policy adapter for ``wangzc9865/SeeNav-Agent``."""

    def __init__(self, name: str, runner: SeeNavRunner):
        super().__init__(
            VLAPolicyConfig(
                name=name,
                action_dim=2,
                action_horizon=SEENAV_MAX_PLAN_ACTIONS,
                default_num_steps=1,
                dtype="bfloat16",
            )
        )
        self.runner = runner
        self.model = runner.model
        self.profile = "seenav"
        self.cuda_graph_requested = False
        self.cuda_graph_enabled = False
        self._decoder = SeeNavDecoder(self)

    @property
    def is_recurrent(self) -> bool:
        return True

    @property
    def manages_cuda_graph(self) -> bool:
        # Keep the standard navigation-engine default usable while SeeNav remains eager.
        return True

    def configure_runtime(self, *, use_cuda_graph: bool) -> None:
        self.cuda_graph_requested = bool(use_cuda_graph)
        self.cuda_graph_enabled = False

    @property
    def decoder(self) -> SeeNavDecoder:
        return self._decoder

    def collate(self, observations, request_ids):
        if len(observations) != 1 or len(request_ids) != 1:
            raise ValueError(f"{self.config.name} requires batch size 1")
        return SeeNavBatch(list(observations), list(request_ids))

    def pad(self, batch, target_batch_size):
        if target_batch_size != 1:
            raise ValueError("SeeNav does not support batch padding")
        return batch

    def encode_prefix(self, batch, memory: MemoryState | None = None):
        if batch.batch_size != 1:
            raise ValueError("SeeNav requires batch size 1")
        observation = batch.observations[0]
        if not observation.instruction:
            raise ValueError("SeeNav navigation instruction is required")
        if observation.images.ndim != 4 or observation.images.shape[0] < 2:
            raise ValueError("SeeNav requires first-person and overhead views")
        if memory is not None and not isinstance(memory, SeeNavMemory):
            raise TypeError(f"SeeNav memory must be SeeNavMemory, got {type(memory).__name__}")
        return SeeNavPrefix(observation, memory or SeeNavMemory())


def _build(
    name,
    checkpoint,
    max_new_tokens,
    overrides,
    *,
    image_concat: bool = True,
    history_window: int = 4,
    load_device: str | None = None,
):
    if overrides:
        raise ValueError(f"unknown {name} overrides: {sorted(overrides)}")
    if checkpoint is None:
        raise ValueError(f"{name} requires a local checkpoint")
    if not Path(checkpoint).exists():
        raise ValueError(f"checkpoint must be local: {checkpoint}")
    return SeeNavPolicy(
        name,
        SeeNavRunner(
            checkpoint,
            max_new_tokens=max_new_tokens,
            image_concat=image_concat,
            history_window=history_window,
            load_device=load_device,
        ),
    )


@register_policy("seenav")
def build_seenav(
    checkpoint=None,
    *,
    max_new_tokens=512,
    image_concat=True,
    history_window=4,
    compile_backend="none",
    tensor_parallel_size: int = 1,
    load_device: str | None = None,
    tensor_parallel_group=None,
    **overrides,
):
    """Build SeeNav with eager B=1 execution."""

    del tensor_parallel_group
    if compile_backend != "none":
        raise ValueError("SeeNav compile_backend must be 'none'")
    if tensor_parallel_size != 1:
        raise ValueError("SeeNav only supports tensor_parallel_size=1")
    return _build(
        "seenav",
        checkpoint,
        max_new_tokens,
        overrides,
        image_concat=image_concat,
        history_window=history_window,
        load_device=load_device,
    )


__all__ = ["SeeNavDecoder", "SeeNavPolicy", "build_seenav"]
