"""SeeNav dual-view prompt construction and image conversion."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ...types import Observation
from .contract import (
    SEENAV_ACTION_NAMES,
    SEENAV_MESSAGE_WINDOW,
    SeeNavMemory,
)

SEENAV_SYSTEM_PROMPT = """You are a robot operating in a home. You can do various tasks and output a sequence of actions to accomplish a given task with images of your status.

The available action id (0 ~ 7) and action names are:
{actions}

Strategy:
1. Locate the target object and describe its spatial location from both views.
2. Use forward and right/left movement as the main strategy, reasoning about obstacles.
3. Focus on moving closer to the target; address an invalid action when it blocks progress.
4. Use rotation or camera tilt sparingly, only when the target is not visible.
5. Do not complete the task until no action can bring the agent closer to the target.
6. Do not repeat an invalid action unless a rotation has already been performed.
"""

SEENAV_JSON_TEMPLATE = """Output only compact JSON with the executable action ids:
{"actions":[0,2]}
The actions field must be a non-empty list of 1 to 8 integers. Each integer must be
between 0 and 7 and must use the action id mapping above. Do not output reasoning,
descriptions, action names, markdown code fences, or any text outside this JSON object.
"""


def tensor_to_seenav_pil(tensor: torch.Tensor):
    """Convert one RGB float tensor in ``[0, 1]`` to a detached RGB PIL image."""

    from PIL import Image

    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError(f"SeeNav images must be [3,H,W], got {tuple(tensor.shape)}")
    if not tensor.is_floating_point():
        raise TypeError("SeeNav images must be floating point tensors in [0, 1]")
    value = tensor.detach().cpu()
    if not torch.isfinite(value).all() or value.min().item() < 0 or value.max().item() > 1:
        raise ValueError("SeeNav images must contain finite values in [0, 1]")
    array = (value.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def concat_seenav_views(bev, fv):
    """Concatenate BEV then first-person view, matching the official prompt."""

    from PIL import Image

    if bev.height != fv.height:
        bev = bev.resize((bev.width, fv.height), Image.Resampling.LANCZOS)
    combined = Image.new("RGB", (bev.width + fv.width, fv.height))
    combined.paste(bev, (0, 0))
    combined.paste(fv, (bev.width, 0))
    return combined


class SeeNavProcessingRuntime:
    """Prompt and processor helpers shared by the local runner and CPU tests."""

    image_concat: bool
    history_window: int

    def _action_listing(self) -> str:
        return "\n".join(f"action id {index}: {name}" for index, name in enumerate(SEENAV_ACTION_NAMES))

    def build_prompt(self, observation: Observation, memory: SeeNavMemory) -> str:
        instruction = (observation.instruction or "").rstrip(".")
        if not instruction:
            raise ValueError("SeeNav navigation instruction is required")
        history = list(memory.turns[-self.history_window :])
        history_text = ""
        if history:
            history_text = "\n\nThe action history:\n"
            for step, turn in enumerate(history):
                names = ", ".join(SEENAV_ACTION_NAMES[action_id] for action_id in turn.action_ids)
                history_text += f"Step {step}, action ids {list(turn.action_ids)} ({names})"
                history_text += "\n"
        feedback = observation.metadata.get("env_feedback")
        if feedback:
            history_text += f"Latest environment feedback: {feedback}\n"
        phase = (
            "Aim for about 1-2 actions in this step."
            if not history
            else "Aim for about 5-6 actions in this step."
        )
        view_description = (
            "The input is one image concatenating the overhead view on the left and the first-person view on the right. "
            "The overhead view shows the agent position and orientation. The colored circle marks the agent, "
            "and the green orientation arrow matches the first-person view. Red target boxes and navigation "
            "arrows, when present, identify the object to reach."
            if self.image_concat
            else "The input contains the first-person view as the first image and the overhead view as the second image. "
            "The overhead view shows the agent position and orientation. Red target boxes and navigation arrows, "
            "when present, identify the object to reach."
        )
        return (
            f"Now the human instruction is: {instruction}.\n\n"
            f"{view_description} Plan using both views.\n"
            f"{phase}\n"
            f"{history_text}\n"
            f"{SEENAV_JSON_TEMPLATE}"
        )

    def _views(self, observation: Observation) -> tuple[torch.Tensor, ...]:
        if observation.images.ndim != 4 or observation.images.shape[0] < 2:
            raise ValueError("SeeNav requires two views in Observation.images: first-person and overhead")
        return (observation.images[0], observation.images[1])

    def build_messages(
        self, observation: Observation, memory: SeeNavMemory
    ) -> tuple[list[dict[str, Any]], list[Any], str]:
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SEENAV_SYSTEM_PROMPT.format(actions=self._action_listing()),
                    }
                ],
            }
        ]
        images: list[Any] = []
        message_window = min(self.history_window, SEENAV_MESSAGE_WINDOW - 1)
        for turn in memory.turns[-message_window:]:
            old_views = [tensor_to_seenav_pil(view) for view in turn.views]
            if self.image_concat:
                old_views = [concat_seenav_views(old_views[1], old_views[0])]
            images.extend(old_views)
            messages.append(
                {
                    "role": "user",
                    "content": [{"type": "image"} for _ in old_views]
                    + [{"type": "text", "text": turn.prompt}],
                }
            )
            messages.append({"role": "assistant", "content": [{"type": "text", "text": turn.response}]})
        views = self._views(observation)
        current_images = [tensor_to_seenav_pil(view) for view in views]
        if self.image_concat:
            current_images = [concat_seenav_views(current_images[1], current_images[0])]
        images.extend(current_images)
        prompt = self.build_prompt(observation, memory)
        messages.append(
            {
                "role": "user",
                "content": [{"type": "image"} for _ in current_images] + [{"type": "text", "text": prompt}],
            }
        )
        return messages, images, prompt

    def build_messages_batch(
        self,
        observations: list[Observation],
        memories: list[SeeNavMemory],
    ) -> tuple[list[str], list[list[Any]], list[str]]:
        """Build independently ordered Qwen text/image rows for a recurrent batch.

        Qwen's processor accepts one image list per text row.  Keeping the rows
        nested here prevents history images from one session being associated with
        another session when their history lengths differ.
        """

        if len(observations) != len(memories):
            raise ValueError("SeeNav observations and memories must have identical lengths")
        texts: list[str] = []
        image_batches: list[list[Any]] = []
        prompts: list[str] = []
        for observation, memory in zip(observations, memories, strict=True):
            messages, images, prompt = self.build_messages(observation, memory)
            texts.append(
                self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                if hasattr(self, "processor")
                else prompt
            )
            image_batches.append(images)
            prompts.append(prompt)
        return texts, image_batches, prompts


__all__ = [
    "SEENAV_JSON_TEMPLATE",
    "SEENAV_SYSTEM_PROMPT",
    "SeeNavProcessingRuntime",
    "concat_seenav_views",
    "tensor_to_seenav_pil",
]
