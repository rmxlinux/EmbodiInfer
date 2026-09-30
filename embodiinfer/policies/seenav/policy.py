"""SeeNav-Agent policy, decoder, and factory registration."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import torch

from ...types import DecodeTrace, Observation
from ..base import MemoryState, PolicyBatch, VLAPolicy
from ..config import VLAPolicyConfig
from ..decoder import BatchedAutoregressiveDecoder, DecodeResult
from ..factory import register_policy
from .contract import (
    SEENAV_MAX_PLAN_ACTIONS,
    SEENAV_MAX_BATCH_SIZE,
    SeeNavBatch,
    SeeNavBatchPrefix,
    SeeNavMemory,
    SeeNavPrefix,
    SeeNavTurn,
    parse_seenav_actions,
)
from .runner import SeeNavRunner


class SeeNavDecoder(BatchedAutoregressiveDecoder):
    """Eager JSON generation for independent recurrent SeeNav sessions."""

    def __init__(self, policy: SeeNavPolicy):
        self.policy = policy

    def decode(
        self,
        state: torch.Tensor | None,
        prefix: SeeNavPrefix,
        num_steps: int,
        bucket: int,
        graphs: object | None,
        *,
        generator: torch.Generator | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> DecodeResult:
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

    def decode_batch(
        self,
        state: torch.Tensor | None,
        prefix: SeeNavBatchPrefix,
        num_steps: int,
        bucket: int,
        graphs: object | None,
        *,
        generator: torch.Generator | None = None,
        cancelled: list[Callable[[], bool]] | None = None,
    ) -> list[DecodeResult]:
        del state, num_steps, bucket, graphs, generator
        if cancelled is not None and any(check() for check in cancelled):
            from ...exceptions import SessionCancelledError

            raise SessionCancelledError("navigation batch cancelled")
        if not isinstance(prefix, SeeNavBatchPrefix):
            raise TypeError(f"SeeNav batch decoder requires SeeNavBatchPrefix, got {type(prefix).__name__}")
        generated = self.policy.runner.infer_batch(
            list(prefix.observations),
            list(prefix.memories),
        )
        if cancelled is not None and any(check() for check in cancelled):
            from ...exceptions import SessionCancelledError

            raise SessionCancelledError("navigation batch cancelled")
        if len(generated) != prefix.batch_size:
            raise RuntimeError(
                f"SeeNav batch runner returned {len(generated)} rows for batch size {prefix.batch_size}"
            )
        results: list[DecodeResult] = []
        for observation, memory, (text, token_ids, entropies) in zip(
            prefix.observations,
            prefix.memories,
            generated,
            strict=True,
        ):
            actions = parse_seenav_actions(text)
            prompt = self.policy.runner.build_prompt(observation, memory)
            views = tuple(view.detach().cpu().clone() for view in observation.images[:2])
            turns = (*memory.turns, SeeNavTurn(views, prompt, text, tuple(int(row[0]) for row in actions)))
            next_memory = SeeNavMemory(turns=turns[-4:])
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
            results.append(
                DecodeResult(
                    actions=actions.unsqueeze(0),
                    next_memory=next_memory,
                    traces=[trace],
                )
            )
        return results


class SeeNavPolicy(VLAPolicy):
    """Eager recurrent SeeNav adapter supporting independent batches up to B=8."""

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
    def supports_recurrent_batch(self) -> bool:
        return True

    @property
    def max_recurrent_batch_size(self) -> int:
        return SEENAV_MAX_BATCH_SIZE

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

    def collate(self, observations: list[Observation], request_ids: list[str]) -> SeeNavBatch:
        return SeeNavBatch(list(observations), list(request_ids))

    def pad(self, batch: PolicyBatch, target_batch_size: int) -> PolicyBatch:
        if target_batch_size != batch.batch_size:
            raise ValueError("SeeNav recurrent batches cannot be padded with synthetic sessions")
        return batch

    def encode_prefix(
        self, batch: SeeNavBatch, memory: MemoryState | None = None
    ) -> SeeNavPrefix:
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

    def encode_prefix_batch(
        self, batch: SeeNavBatch, memories: Sequence[MemoryState | None]
    ) -> SeeNavBatchPrefix:
        if batch.batch_size != len(memories):
            raise ValueError("SeeNav batch and memory rows must have identical lengths")
        if not 1 <= batch.batch_size <= SEENAV_MAX_BATCH_SIZE:
            raise ValueError(f"SeeNav batch size must be between 1 and {SEENAV_MAX_BATCH_SIZE}")
        normalized: list[SeeNavMemory] = []
        for observation, memory in zip(batch.observations, memories, strict=True):
            if not observation.instruction:
                raise ValueError("SeeNav navigation instruction is required")
            if observation.images.ndim != 4 or observation.images.shape[0] < 2:
                raise ValueError("SeeNav requires first-person and overhead views")
            if memory is not None and not isinstance(memory, SeeNavMemory):
                raise TypeError(f"SeeNav memory must be SeeNavMemory, got {type(memory).__name__}")
            normalized.append(memory or SeeNavMemory())
        return SeeNavBatchPrefix(tuple(batch.observations), tuple(normalized), batch.batch_size)


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
    max_new_tokens=64,
    image_concat=True,
    history_window=4,
    compile_backend="none",
    tensor_parallel_size: int = 1,
    load_device: str | None = None,
    tensor_parallel_group=None,
    **overrides,
):
    """Build SeeNav with eager independent-session batches up to B=8."""

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
