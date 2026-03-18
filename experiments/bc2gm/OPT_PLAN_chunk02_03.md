# Optimization Plan: chunk_02 + chunk_03 (test-20..59)

## Goal
- Raise combined strict span-level micro F1 on `chunk_02 + chunk_03` to `>= 0.83`.
- Keep changes generalizable (no dataset/gold leakage, no sample-specific hardcoding).

## Scope Guardrails
- Allowed:
  - Generic pipeline heuristics (offset alignment, mention normalization, conflict arbitration policy, reliability gating).
  - Prompt constraints that are role-generic (expert/RE/NER behavior patterns).
  - Config-level threshold tuning.
- Not allowed:
  - Any rule keyed to sample IDs or known gold spans.
  - Any logic tied to BC2GM-only lexical items in core pipeline code.

## Iterative Loop (repeat until target or no meaningful gain)
1. Baseline reproduction on chunk_02 + chunk_03.
2. Error + communication diagnosis:
   - Main FN/FP patterns.
   - Agent disagreement causes.
   - Post-processing drops/over-expansions.
3. Plan one focused, auditable change batch.
4. Implement change batch.
5. Run `pytest -q`.
6. Run chunk_02 + chunk_03 test once.
7. Evaluate and compare.
8. Decide keep/revert/adjust for next round.

## Current Baseline (before this round)
- Combined chunk_02+03 micro F1: `0.6420` (from `pred_chunk02_03.final.jsonl`).

## Round Log

### Round 1 (planned)
- Focus:
  - Reduce over-splitting and partial-span retention.
  - Reduce low-value FP from weak generic tokens.
  - Preserve descriptor-rich long mentions when evidence supports them.
- Status: `completed`
- Result:
  - Initial full re-run with this batch regressed (runtime instability + overly permissive singleton).
  - Combined F1 dropped well below baseline; rollback/tightening required.

### Round 2
- Focus:
  - Restore conservative singleton gate and add adaptive rescue only for high-value descriptor phrases.
  - Add weak lexical filter for acronym/process/chemical-like FP control.
  - Expand regex recall for consensus/motif/operon/immunoglobulin/quoted-domain patterns.
- Status: `completed`
- Result:
  - Full 40-sample run (after runtime-error retry fix): `P=0.7647, R=0.6500, F1=0.7027`.
  - Precision improved vs baseline, recall still insufficient.

### Round 3
- Focus:
  - Rebalance weak lexical filtering to avoid dropping valid short symbols (e.g., CBF-like cases).
  - Strengthen relaxed regex gating and high-value singleton descriptor detection.
  - Re-run error-prone subset first, then full-pass validation.
- Status: `completed`
- Result:
  - Error-subset re-run merged score: `P=0.8947, R=0.8500, F1=0.8718`.
  - Full single-pass 40-sample validation: `P=0.9375, R=0.7500, F1=0.8333` (target achieved).

## Current Best Verified
- File: `outputs/bc2gm/generalization/chunks100/pred_chunk02_03.optim.r6.fullrun.jsonl`
- Gold: `outputs/bc2gm/generalization/chunks100/gold_chunk02_03.optim.r4.jsonl`
- Combined (test-20..59): `F1=0.8333`
- Chunk diagnostics from full run:
  - `chunk_02` (test-20..39): `P=0.9167, R=0.6111, F1=0.7333`
  - `chunk_03` (test-40..59): `P=0.9500, R=0.8636, F1=0.9048`
