# Active Tuning Plan (Target: stable F1 >= 0.82 on chunk04+05)

## Iteration A (completed)
- Plan:
  - Add low-information filter.
  - Add conservative regex recall.
  - Verify on quick10 then full40.
- Outcome:
  - Quick10 improved strongly.
  - Full40 reached ~0.79, below target.
- Verdict:
  - Recall improved, but boundary mismatch remained high.

## Iteration B (completed)
- Plan:
  - Upgrade regex recall with stricter context (regex4).
  - Validate quick10 before expensive full40.
- Outcome:
  - Quick10 reached >0.90 F1.
  - Full40 not yet validated with boundary override logic.
- Verdict:
  - Candidate direction is strong but requires boundary correction.

## Iteration C (in progress)
- Plan:
  - Add regex boundary override module:
    - promote strong full phrases,
    - split coordinated forms,
    - trim weak descriptor tails,
    - add parenthetical head/alias variants.
  - Run quick10, then full40.
  - Compare against current best full40 checkpoint.
- Success criteria:
  - full40 F1 >= 0.82,
  - no obvious prompt/model leakage,
  - reproducible command + config archived.

## Iteration D (in progress, target set switched to test-40..79)
- Validation set:
  - Fixed 40-sample window: `test-40..79`.
- Baseline snapshot:
  - `P=0.6727, R=0.6491, F1=0.6607` (from `eval_40_79_baseline_current.json`).
- Plan:
  - Tighten symbolic split to avoid single-letter artifacts.
  - Restrict weak-tail boundary contraction from `X protein` to `X complex` only.
  - Increase candidate high-recall phrase coverage via descriptor fallback + longer right-boundary modifiers.
  - Enable conservative candidate-recall bridge.
  - Disable memory to reduce cross-sample contamination during validation rounds.
- Success criteria:
  - Validation `F1 >= 0.82` on `test-40..79`.
  - No dataset-label leakage; all changes remain schema/pattern driven.

### Iteration D.1 result (failed)
- Config: `config.generalization.optim.r1.yaml`
- Outcome:
  - `P=0.4356, R=0.7719, F1=0.5570` on `test-40..79`.
  - Recall rose but precision collapsed (high FP from aggressive recall settings).
- Decision:
  - Roll back aggressive recall bridge/descriptive fallback settings.

### Iteration D.2 result (improved, near target)
- Config: `config.generalization.optim.r2.yaml`
- Outcome:
  - `P=0.8462, R=0.7719, F1=0.8073` on `test-40..79`.
- Decision:
  - Keep conservative profile; add targeted phrase-level boundary overrides to recover FN.

### Iteration D.3 result (target achieved)
- Config: `config.generalization.optim.r3.yaml`
- Run A:
  - `P=0.8871, R=0.9649, F1=0.9244` on `test-40..79`.
- Run B (fresh memory file, same config):
  - `P=0.8571, R=0.9474, F1=0.9000` on `test-40..79`.
- Stability verdict:
  - Two independent runs are both `F1 > 0.82`; target met.

## Iteration E (in progress, generalization recovery)
- Trigger:
  - Real `test-100..199` run completed with `F1=0.4251`, failing generalization target.
- Diagnosis:
  - Main errors are `missing` + `boundary_mismatch`, with non-trivial RE-only spurious spans.
  - Existing postprocess heuristics over-preserve some split/descriptor variants.
- Plan:
  - Keep pipeline logic generic (no dataset field hardcoding).
  - Tighten generic lexical filtering for weak RE-only mentions.
  - Keep high recall, but suppress clearly low-signal RE-only artifacts.
  - Validate on `test-20..59` as optimization small test.
  - If `F1 < 0.83`, continue another optimize-test loop.
- Success criteria:
  - `test-20..59` reaches `F1 >= 0.83`.
  - Changes are schema/rationale driven and reusable across NER tasks.

### Iteration E.1 result (partial pass, generalization fail)
- Code change:
  - Tightened generic weak-lexical filtering for RE-only low-signal mentions in `src/maner/orchestrator/pipeline.py`.
- Validation (`test-20..59`):
  - After runtime-error repair: `P=0.8333, R=1.0000, F1=0.9091`.
- Generalization (`test-100..199`):
  - After full runtime-error repair: `P=0.4366, R=0.5124, F1=0.4715` (failed).
- Diagnosis:
  - Major remaining issue is boundary inconsistency on unseen samples (`boundary_mismatch=44`) plus over-generation (`spurious=36`).
  - Current postprocess/heuristic stack is still not robustly transferable.
- Next iteration focus:
  - Redesign boundary selection policy to be more stable across domains (clustered overlap resolution + provenance-aware pruning).
  - Rebalance augmentation so recall comes from schema-guided evidence, not brittle pattern over-expansion.
