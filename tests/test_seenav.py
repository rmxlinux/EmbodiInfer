from __future__ import annotations

import os

import pytest
import torch

from embodiinfer import EngineConfig, Observation
from embodiinfer.engine import EngineCore
from embodiinfer.policies.seenav.contract import (
    SEENAV_ACTION_NAMES,
    SeeNavMemory,
    SeeNavOutputError,
    parse_seenav_actions,
)
from embodiinfer.policies.seenav.policy import SeeNavPolicy
from embodiinfer.policies.seenav.processing import SeeNavProcessingRuntime
from embodiinfer.types import SessionKey


def observation() -> Observation:
    return Observation(
        images=torch.zeros(2, 3, 8, 8),
        state=torch.empty(0),
        instruction_tokens=torch.empty(0, dtype=torch.long),
        instruction="find the mug",
        metadata={"env_feedback": "the previous move succeeded"},
    )


class FakeRunner:
    image_concat = True
    history_window = 4

    def __init__(self) -> None:
        self.model = torch.nn.Linear(1, 1)
        self.prompts: list[str] = []

    def build_prompt(self, obs, memory):
        runtime = SeeNavProcessingRuntime()
        runtime.image_concat = self.image_concat
        runtime.history_window = self.history_window
        prompt = runtime.build_prompt(obs, memory)
        self.prompts.append(prompt)
        return prompt

    def infer(self, obs, memory):
        del obs, memory
        text = '{"visual_state_description":"ok","reasoning_and_reflection":"ok","language_plan":"forward","executable_plan":[{"action_id":0,"action_name":"Move forward by 0.25"}]}'
        return text, torch.tensor([11, 12]), []


class InvalidRunner(FakeRunner):
    def infer(self, obs, memory):
        del obs, memory
        return '{"executable_plan": []}', torch.tensor([13]), []


def test_seenav_parser_accepts_json_and_code_fence() -> None:
    text = """```json
    {"executable_plan": [{"action_id": 0, "action_name": "Move forward by 0.25"}, {"action_id": 5, "action_name": "Rotate left by 90 degrees"}]}
    ```"""
    assert parse_seenav_actions(text).tolist() == [[0.0, 0.0], [5.0, 0.0]]


@pytest.mark.parametrize(
    "text",
    [
        "",
        "not json",
        '{"executable_plan": []}',
        '{"executable_plan": [{"action_id": 8, "action_name": "bad"}]}',
        '{"executable_plan": [{"action_id": 0}]}',
    ],
)
def test_seenav_parser_rejects_invalid_plans(text: str) -> None:
    with pytest.raises(SeeNavOutputError):
        parse_seenav_actions(text)


def test_seenav_prompt_contains_official_action_space_and_history() -> None:
    runtime = SeeNavProcessingRuntime()
    runtime.image_concat = True
    runtime.history_window = 4
    first = observation()
    prompt = runtime.build_prompt(first, SeeNavMemory())
    messages, _, _ = runtime.build_messages(first, SeeNavMemory())
    assert "action id 0: Move forward by 0.25" in messages[0]["content"][0]["text"]
    assert "concatenating the overhead view on the left" in prompt


def test_seenav_build_messages_preserves_dual_view_order() -> None:
    runtime = SeeNavProcessingRuntime()
    runtime.image_concat = True
    runtime.history_window = 4
    messages, images, _ = runtime.build_messages(observation(), SeeNavMemory())
    assert messages[-1]["content"][0]["type"] == "image"
    assert len(images) == 1
    assert images[0].size == (16, 8)


def test_seenav_policy_is_explicit_b1_and_engine_commits_memory() -> None:
    runner = FakeRunner()
    policy = SeeNavPolicy("seenav", runner)
    obs = observation()
    batch = policy.collate([obs], ["request"])
    core = EngineCore(policy, EngineConfig(device="cpu", max_batch_size=1))
    output = core.execute(batch, session_ids=[SessionKey("env", "episode")])[0]
    assert output.actions.tolist() == [[0.0, 0.0]]
    assert output.trace is not None
    assert output.trace.parsed_actions == [[0.0, 0.0]]
    assert output.trace.meta["runtime_mode"] == "eager"
    assert output.trace.meta["cuda_graph_confirmed"] is False
    assert core.has_session_state()
    with pytest.raises(ValueError, match="batch size 1"):
        policy.collate([obs, obs], ["a", "b"])
    with pytest.raises(ValueError, match="batch padding"):
        policy.pad(batch, 2)


def test_seenav_parser_failure_rolls_back_the_session() -> None:
    policy = SeeNavPolicy("seenav", InvalidRunner())
    batch = policy.collate([observation()], ["request"])
    core = EngineCore(policy, EngineConfig(device="cpu", max_batch_size=1))
    with pytest.raises(SeeNavOutputError):
        core.execute(batch, session_ids=[SessionKey("env", "episode")])
    assert not core.has_session_state()


def test_seenav_memory_only_allows_b1_operations() -> None:
    memory = SeeNavMemory()
    assert memory.expand(1) is memory
    with pytest.raises(ValueError, match="B=1"):
        memory.expand(2)
    with pytest.raises(ValueError, match="compact"):
        memory.compact([1])


def test_seenav_action_names_are_the_eight_upstream_actions() -> None:
    assert len(SEENAV_ACTION_NAMES) == 8
    assert SEENAV_ACTION_NAMES[0].startswith("Move forward")


@pytest.mark.seenav
def test_seenav_real_checkpoint_smoke() -> None:
    checkpoint = os.environ.get("SEENAV_CHECKPOINT")
    if not checkpoint:
        pytest.skip("set SEENAV_CHECKPOINT to run the weight-dependent smoke test")
    from embodiinfer import make_policy

    policy = make_policy("seenav", checkpoint=checkpoint, load_device="cpu", max_new_tokens=8)
    assert policy.runner.revision == "b1024343452d2cf42bd23b4dc6e8efe01313b549"
