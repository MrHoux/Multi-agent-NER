# Generic Optimization Plan (20-59 -> then 100-199)

## Objective
- Reach strict span-level micro F1 > 0.83 on test-20..59.
- Then run one-pass evaluation on test-100..199 for generalization check.

## Hard Constraints
- No sample-id rules, no gold leakage, no dataset-field-specific recognizers in pipeline.
- Domain knowledge is allowed only through Expert/RAG prompts and schema-conditioned reasoning.
- Core pipeline logic must stay schema-driven / task-generic.

## Iteration Protocol
1. Baseline run on test-20..59 with current genericized code.
2. Error + communication analysis:
   - boundary mismatch
   - spurious mentions
   - missing mentions
   - agent disagreement / abstain behavior
3. Apply one focused change batch (prompt/config/multi-agent policy).
4. Run `pytest -q`.
5. Re-run test-20..59.
6. Compare metrics and keep/reject changes.
7. Repeat until F1 > 0.83.
8. Freeze config and run test-100..199 in 5 chunks with live per-chunk stats.
9. If 100..199 degrades abnormally, restart loop from step 2 with stronger generality constraints.

## Allowed Change Types
- Prompt redesign per agent (candidate/expert/re/ner/adjudicator/debate/verifier).
- Agent interaction policy (conflict triage, abstain rules, debate trigger).
- Schema-driven generic heuristics and confidence calibration.
- Memory gating and retrieval policy.

## Disallowed Change Types
- Any hardcoded lexical rule aimed at specific sample content.
- Any direct dependence on dataset-specific IDs/offsets/special fields beyond generic id/text/gold_mentions schema.
- Any rule that bakes in observed gold answers.

## Progress Log
- Status: in_progress

### Baseline (current code before this round)
- Config: `experiments/bc2gm/config.generalization.deepseek.yaml` (pre-round edits)
- Data: `chunk_02 + chunk_03` (`test-20..59`)
- Result:
  - `20-39`: F1 `0.5556` (P `0.5556`, R `0.5556`)
  - `40-59`: F1 `0.3333` (P `0.3500`, R `0.3182`)
  - `20-59`: F1 `0.4359` (P `0.4474`, R `0.4250`)

### Round 1
- Change batch:
  - stricter singleton entity-surface gate in `adjudicator_agent.py`
  - weak-lexical filter refinement (mixed-case acronym retention + function-word drop)
  - unified config tuning (more fallback/bridge)
  - prompt hard constraints for non-entity generic words
- Result:
  - `20-59`: F1 `0.3846` (degraded, high FP from candidate recall bridge)
- Decision: reject this config mix; keep code changes, retune config.

### Round 2
- Change batch:
  - disable candidate recall bridge
  - enable regex pattern recall
  - add quantifier-led generic phrase drop in weak lexical filter
- Result:
  - `20-39`: F1 `0.6897`
  - `40-59`: F1 `0.5789`
  - `20-59`: F1 `0.6269`
- Decision: keep (large precision gain), continue recall repair.

### Round 3
- Change batch:
  - schema description enriched with biomedical named-form guidance
  - candidate fallback descriptor terms added via config
  - weak lexical acronym threshold relaxed
  - low-information filter disabled (to prevent over-drop)
- Result:
  - `20-59`: F1 `0.6250` (flat vs round 2)
- Decision: keep partial, continue.

### Round 4 (control with historical robust config family)
- Config: `experiments/bc2gm/config.generalization.optim.r3.yaml`
- Result:
  - `20-39`: F1 `0.7000`
  - `40-59`: F1 `0.6364`
  - `20-59`: F1 `0.6667` (best in this optimization session so far)

### Round 5
- Change batch:
  - boundary override enabled + descriptor term expansion list
  - list-context fallback disabled
- Result:
  - `20-59`: F1 `0.6000` (degraded)
- Decision: reject.

### Round 6
- Change batch:
  - prompt/schema recall relaxation for lowercase biomedical entities
- Result:
  - `20-59`: F1 `0.5510` (degraded due precision collapse)
- Decision: reject.

### Round 7
- Change batch:
  - add generic candidate superspan promotion (schema-term + entity-like guard)
- Config:
  - `experiments/bc2gm/config.generalization.optim.r3.yaml`
  - overrides: `pipeline.enable_candidate_superspan_promotion=true`, `pipeline.candidate_superspan_max_expand_chars=40`
- Result:
  - `20-39`: F1 `0.8333`
  - `40-59`: F1 `0.6047`
  - `20-59`: F1 `0.7089` (session best)

### Round 8
- Change batch:
  - singleton confidence grace + stricter superspan tail checks
- Result:
  - `20-59`: F1 `0.6420` (degraded)
- Decision: reject.

### Round 9
- Change batch:
  - config-driven regex extra patterns (no hard-coded sample ids/fields)
- Result:
  - `20-59`: F1 `0.6753` (below round 7)
- Decision: partial keep (config-driven capability), retune.

### Round 10
- Change batch:
  - wider regex head fallback + lower acronym singleton gate
- Result:
  - `20-59`: F1 `0.6173` (degraded)
- Decision: reject.

### Round 11
- Change batch:
  - restrict relaxed regex head fallback to extra patterns only
  - restore conservative singleton/acronym thresholds
- Result:
  - `20-39`: F1 `0.7368`
  - `40-59`: F1 `0.6486`
  - `20-59`: F1 `0.6933`
- Decision:
  - not reaching target `0.83`; keep investigating chunk_03 recall bottlenecks.
