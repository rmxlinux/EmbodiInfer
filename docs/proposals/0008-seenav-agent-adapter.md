# 0008 — SeeNav-Agent policy adapter

- Status: Implemented
- Author: EmbodiInfer maintainers
- Date: 2026-09-29

## 1. Summary

Add a policy-layer adapter for `wangzc9865/SeeNav-Agent`, a Qwen2.5-VL-3B
dual-view navigation checkpoint. The adapter uses the existing explicit-session
protocol and eager autoregressive decoder, with a deliberate batch-one contract.

## 2. Motivation and current gap

The policy catalog has Qwen2.5-VL navigation adapters for R2R low-level,
panoramic, NaViDA, and ActiveVLN, but no SeeNav adapter. SeeNav emits a structured
JSON action plan and consumes a first-person/overhead visual prompt, so its loading,
prompt, history, and parser behavior cannot be expressed by the existing single-token
or free-form action adapters.

## 3. Goals and non-goals

Goals are local checkpoint loading, official dual-view prompt construction, explicit
recurrent history, strict action-plan parsing, and B=1 operation through `EngineCore`.
CUDA Graphs, tensor parallelism, RL log-probability, HTTP serving, and environment
or simulator integration are non-goals.

## 4. Design

`embodiinfer.policies.seenav` owns the Qwen processor/model runner, prompt and PIL
conversion, `SeeNavMemory`, and JSON parser. `SeeNavPolicy` exposes the existing
`VLAPolicy` and `AutoregressiveDecoder` contracts. The runner keeps at most four
completed turns, concatenating overhead then first-person views by default. A strict
parser rejects malformed, empty, or out-of-range plans. Reusing the R2R parser was
rejected because it assumes one textual action token and cannot represent SeeNav's
structured plan.

## 5. Model-agnosticism verdict

This is policy-layer code. It depends on public `VLAPolicy`, `ActionDecoder`,
`Observation`, and `SessionKey` contracts; SeeNav-specific action IDs and Qwen
processor behavior remain inside the adapter.

## 6. Losslessness and precision criterion

The reference is the upstream local Qwen2.5-VL call with `temperature=0.001` and
greedy generation. The adapter compares generated token IDs and parsed action IDs on
the same CPU RGB inputs, seed, checkpoint revision, and `max_new_tokens`; no numerical
tolerance is substituted for token or parser mismatches.

## 7. Implementation plan

Add `embodiinfer/policies/seenav/{contract,processing,runner,policy}.py`, compatibility
exports, lazy registry entries, model documentation, and the qwen25-vln installation
note. Keep the existing public imports and do not alter the engine.

## 8. Test plan

CPU tests cover strict JSON parsing, dual-view prompt construction, memory transactions,
batch-size rejection, and a fake-runner `EngineCore` B=1 execution. A checkpoint test
is environment-gated and records the exact SeeNav revision. GPU smoke/performance tests
use the local machine before any 4090 claim.

## 9. Benchmark plan

Follow the repository's navigation benchmark convention: batch size 1, BF16, fixed
seed, separate warmup, decoded CPU RGB input through CPU action output, E2E and model
interval timing, and explicit episode reset. Report hardware, checkpoint revision,
software versions, output counts, and memory peaks. Do not compare SeeNav throughput
with StreamVLN without matching prompt and output workloads.

## 10. Risks and limitations

The model is trained for the EmbodiedBench dual-view prompt and may require marked
views from the downstream environment. A plan can contain a variable number of action
rows, so downstream execution owns action horizon semantics. Batching, graph capture,
and simulator success metrics remain unsupported.
