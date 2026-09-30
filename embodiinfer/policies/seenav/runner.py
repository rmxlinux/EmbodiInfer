"""Local Hugging Face runner for SeeNav-Agent."""

from __future__ import annotations

from pathlib import Path

import torch

from .contract import SEENAV_REVISION, SeeNavMemory
from .processing import SeeNavProcessingRuntime


class SeeNavRunner(SeeNavProcessingRuntime):
    """Run the official Qwen2.5-VL SeeNav checkpoint one episode at a time."""

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

    def _encode(self, observation, memory: SeeNavMemory):
        messages, images, prompt = self.build_messages(observation, memory)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = self.processor(text=[text], images=images, padding=True, return_tensors="pt")
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        encoded = {
            key: value.to(device=device, dtype=dtype) if value.is_floating_point() else value.to(device)
            for key, value in encoded.items()
        }
        text_config = getattr(self.model.config, "text_config", self.model.config)
        context_limit = int(getattr(text_config, "max_position_embeddings", 0))
        if context_limit and encoded["input_ids"].shape[1] > context_limit:
            raise ValueError(
                f"SeeNav prompt has {encoded['input_ids'].shape[1]} tokens, above {context_limit}"
            )
        return encoded, prompt

    @torch.inference_mode()
    def infer(self, observation, memory: SeeNavMemory):
        encoded, _ = self._encode(observation, memory)
        generated = self.model.generate(
            **encoded,
            do_sample=False,
            temperature=0.001,
            max_new_tokens=self.max_new_tokens,
        )
        prompt_length = int(encoded["input_ids"].shape[1])
        token_ids = generated[:, prompt_length:]
        text = self.processor.batch_decode(
            token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return text.strip(), token_ids[0].detach(), []


__all__ = ["SeeNavRunner"]
