"""Local Hugging Face runner for SeeNav-Agent."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

from .contract import SEENAV_MAX_BATCH_SIZE, SEENAV_REVISION, SeeNavMemory
from .processing import SeeNavProcessingRuntime

SeeNavInference = tuple[str, torch.Tensor, list[float]]


class SeeNavRunner(SeeNavProcessingRuntime):
    """Run the official Qwen2.5-VL SeeNav checkpoint with eager B<=8 batching."""

    def __init__(
        self,
        checkpoint: str,
        *,
        max_new_tokens: int = 64,
        image_concat: bool = True,
        history_window: int = 4,
        load_device: str | None = None,
    ) -> None:
        if max_new_tokens < 1:
            raise ValueError("SeeNav max_new_tokens must be positive")
        if not 1 <= history_window <= 4:
            raise ValueError("SeeNav history_window must be between 1 and 4")
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise ValueError(f"checkpoint must be a local SeeNav snapshot: {checkpoint}")
        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "SeeNav requires the isolated Qwen2.5-VL runtime; install the qwen25-vln dependency group"
            ) from exc

        self.checkpoint = str(checkpoint_path)
        self.revision = SEENAV_REVISION
        self.max_new_tokens = max_new_tokens
        self.image_concat = image_concat
        self.history_window = history_window
        self.processor = AutoProcessor.from_pretrained(self.checkpoint, local_files_only=True)
        self.processor.tokenizer.padding_side = "left"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.checkpoint, local_files_only=True, torch_dtype=torch.bfloat16
        )
        config = self.model.config
        text_config = getattr(config, "text_config", config)
        if config.model_type != "qwen2_5_vl" or int(text_config.hidden_size) != 2048:
            raise ValueError(
                "SeeNav is restricted to Qwen2.5-VL-3B "
                f"(got model_type={config.model_type!r}, hidden_size={text_config.hidden_size})"
            )
        if load_device is not None:
            self.model.to(load_device)

    def _move_encoded(self, encoded: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        return {
            key: value.to(device=device, dtype=dtype) if value.is_floating_point() else value.to(device)
            for key, value in encoded.items()
        }

    def _check_context(self, encoded: dict[str, torch.Tensor]) -> None:
        text_config = getattr(self.model.config, "text_config", self.model.config)
        context_limit = int(getattr(text_config, "max_position_embeddings", 0))
        if context_limit and encoded["input_ids"].shape[1] > context_limit:
            raise ValueError(
                f"SeeNav prompt has {encoded['input_ids'].shape[1]} tokens, above {context_limit}"
            )

    def _encode(self, observation, memory: SeeNavMemory) -> tuple[dict[str, torch.Tensor], str]:
        encoded, prompt = self._encode_batch([observation], [memory])
        return encoded, prompt[0]

    def _encode_batch(
        self, observations: Sequence, memories: Sequence[SeeNavMemory]
    ) -> tuple[dict[str, torch.Tensor], list[str]]:
        if not 1 <= len(observations) <= SEENAV_MAX_BATCH_SIZE:
            raise ValueError(f"SeeNav batch size must be between 1 and {SEENAV_MAX_BATCH_SIZE}")
        if len(observations) != len(memories):
            raise ValueError("SeeNav observations and memories must have identical lengths")
        texts, image_batches, prompts = self.build_messages_batch(observations, memories)
        encoded = self.processor(
            text=texts,
            images=image_batches,
            padding=True,
            return_tensors="pt",
        )
        self._check_context(encoded)
        return self._move_encoded(encoded), prompts

    def _trim_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Remove generation padding while retaining the first EOS token."""

        eos_ids = getattr(self.model.generation_config, "eos_token_id", None)
        if eos_ids is None:
            return token_ids
        eos_set = {int(eos_ids)} if isinstance(eos_ids, int) else {int(value) for value in eos_ids}
        for index, value in enumerate(token_ids.tolist()):
            if int(value) in eos_set:
                return token_ids[: index + 1]
        return token_ids

    @torch.inference_mode()
    def infer(self, observation, memory: SeeNavMemory) -> SeeNavInference:
        results = self.infer_batch([observation], [memory])
        return results[0]

    @torch.inference_mode()
    def infer_batch(
        self, observations: Sequence, memories: Sequence[SeeNavMemory]
    ) -> list[SeeNavInference]:
        """Generate and decode up to eight independent recurrent sessions in one call."""

        encoded, _ = self._encode_batch(observations, memories)
        generated = self.model.generate(
            **encoded,
            do_sample=False,
            temperature=0.001,
            max_new_tokens=self.max_new_tokens,
        )
        prompt_length = int(encoded["input_ids"].shape[1])
        token_rows = [self._trim_tokens(row[prompt_length:].detach()) for row in generated]
        texts = [
            self.processor.batch_decode(
                token_row.unsqueeze(0),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            for token_row in token_rows
        ]
        return [
            (text, token_ids, [])
            for text, token_ids in zip(texts, token_rows, strict=True)
        ]


__all__ = ["SeeNavRunner"]
