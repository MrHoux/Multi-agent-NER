# 60-99 Optimization Plan (Long-Run, Interruptible)

## Goal
- Optimize the multi-agent pipeline on a fixed validation window: `test-60..99`.
- After each optimization round, run exactly one test pass on `60..99` and evaluate strict span-level F1.
- Target: `F1 >= 0.82`.

## Guardrails
- Keep pipeline generic; no dataset/sample-specific hardcoding in `src/`.
- Domain-specific knowledge should stay in expert/RAG prompting/config layers.
- Every code change must be auditable and small-scope.

## Baseline
- Current baseline report:
  - `outputs/bc2gm/generalization/chunks100/eval_60_99.eval60_99.r1.json`
  - overall F1: `0.6301`

## Iteration Loop (repeat until target met)
1. Error analysis:
   - Per-chunk and per-sample TP/FP/FN breakdown.
   - Boundary/type/spurious/missing decomposition.
2. Agent communication analysis:
   - Candidate coverage quality.
   - Expert/RE disagreements and negative-rationale filtering effects.
   - Conflict/debate trigger behavior and adjudicator decisions.
3. Optimization patch (generic):
   - Adjust agent interaction and post-processing logic.
   - Keep changes minimal and testable.
4. Verification:
   - Run `pytest -q`.
5. Small test run (`60..99`, one round):
   - Re-run chunk07..10 once.
   - Merge and evaluate strict F1.
6. Decision:
   - If `F1 >= 0.82`: stop and report.
   - Else: continue loop with next hypothesis.

## Progress Log
- Iteration 0 (baseline): F1=0.6301, target not met.

## Iteration 1 (planned)
- Hypothesis:
  - A large share of FP comes from schema-description noise leaking into regex recall terms and loose acronym recall.
  - A portion of FN comes from occasional sample-level JSON parse failure that turns a sample into empty output.
- Planned generic changes:
  1. Tighten schema descriptor term extraction (drop exclusion/negative clauses and generic noise tokens).
  2. Tighten regex recall head-token and acronym gating (symbolic-token preference, avoid substring cue matching).
  3. Improve JSON parsing robustness with multi-candidate object extraction + minimal normalization.
- Validation:
  - `pytest -q`
  - one pass on `test-60..99`
  - evaluate strict F1 and compare against baseline 0.6301.

## Iteration 1 (result)
- Code changes:
  1. tightened regex recall head-token gating and acronym context token matching.
  2. improved JSON parsing robustness with multi-candidate extraction/fix attempts.
- Test:
  - `python -m pytest -q` -> passed.
- 60-99 run outcome:
  - observed F1 `0.6508`, but run quality invalid due transient network failures.
  - `test-61..69` became runtime-error empty predictions (connection reset), so this score is not a clean algorithmic comparison.

## Iteration 2 (planned)
- Hypothesis:
  - Sample-level hard fail on transient API/network issues is a major instability source and hides true model quality.
- Planned generic changes:
  1. add pipeline sample-level retry with exponential backoff for retryable runtime errors.
  2. keep non-retryable failures behavior unchanged (continue_on_error fallback).
- Validation:
  - `pytest -q`
  - rerun `test-60..99` one round with same config and compare F1 to baseline.

## Iteration 2 (result)
- Code/config changes:
  - pipeline sample-level retry (`sample_max_retries`, backoff).
  - candidate descriptor fallback enabled, span-augmentation length relaxed, regex boundary precision tightened.
  - slash-coordinated mention split added in postprocess.
- Test:
  - `python -m pytest -q` -> passed.
- 60-99 one-round result:
  - `F1 = 0.7538` (runtime_error=0), improved but still below 0.82.
  - Main residual errors: boundary mismatch and existence errors on `80-89` / `90-99`.

## Iteration 3 (planned)
- Hypothesis:
  - Regex/boundary injected mentions are not getting a second normalize/merge pass.
  - Some explicit negative rationales ("excluded per ...") are not filtered.
- Planned generic changes:
  1. add final refine postprocess pass after regex/boundary injections.
  2. extend negative-rationale filter patterns.
  3. cautiously enable candidate-recall bridge for missed candidate spans.
- Validation:
  - `pytest -q`
  - rerun `test-60..99` one round.

## Iteration 3 (result)
- Code/config changes:
  - added final refine pass after regex/boundary injections.
  - extended negative-rationale filter.
  - enabled candidate-recall bridge (conservative).
- Test:
  - `python -m pytest -q` -> passed.
- 60-99 one-round result:
  - `F1 = 0.7006` (runtime_error=0), regression due precision collapse.
  - Root cause: candidate-recall bridge over-triggered (many extra mentions).

## Iteration 4 (planned)
- Hypothesis:
  - Pipeline architecture changes are no longer the dominant bottleneck; model quality is limiting.
- Planned generic changes:
  1. disable candidate-recall bridge regression source.
  2. switch model to `deepseek-reasoner` for stronger structured extraction while keeping same pipeline.
- Validation:
  - rerun `test-60..99` one round and check F1 against 0.82.
