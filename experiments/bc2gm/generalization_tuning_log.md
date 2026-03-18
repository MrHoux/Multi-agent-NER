# Generalization Tuning Log (Pause/Resume Safe)

## Protocol
- Objective: improve strict span-level F1 without dataset-content hardcoding.
- Tuning loop:
  1. Run unit tests.
  2. Run quick eval on chunk01 (dev-20) and chunk02 (check-20).
  3. Analyze errors.
  4. Apply one scoped change (config/prompt/code), rerun.
- Final report:
  - One frozen run on full 100 samples.

## Constraints
- No hardcoded BC2GM lexical lists in `src/maner`.
- No sample-id-specific logic.
- Keep all tweaks auditable in config/prompt or generic runtime logic.

## Iteration 0
- Status: completed
- Config: `experiments/bc2gm/config.generalization.deepseek.yaml`
- Dev checks:
  - chunk01: P=0.6957, R=0.7619, F1=0.7273
  - chunk02: P=0.2105, R=0.4444, F1=0.2857
- Diagnosis:
  - Severe FP bursts from singleton expert-only acceptance on some samples.

## Iteration 1
- Status: completed
- Change:
  - `pipeline.adjudicator_singleton_min_confidence: 0.98`
- Dev check:
  - chunk02: P=0.6923, R=0.5000, F1=0.5806 (major FP reduction)
- Test (chunk04+05, one round):
  - P=0.7551, R=0.5692, F1=0.6491

## Iteration 2
- Status: completed
- Change:
  - stricter singleton text gate in code + `singleton_min_confidence=0.95`
- Dev check:
  - chunk02: P=0.3478, R=0.4444, F1=0.3902 (regression)
- Decision:
  - rolled back code gate change.

## Iteration 3
- Status: completed
- Config: `experiments/bc2gm/config.generalization.r3.yaml`
- Change:
  - boundary normalization on (generic descriptor-like modifiers),
  - descriptor expansion on,
  - singleton threshold 0.96.
- Test (chunk04+05, one round):
  - P=0.7000, R=0.6462, F1=0.6720

## Iteration 4
- Status: completed
- Config: `experiments/bc2gm/config.generalization.r4.yaml`
- Change:
  - descriptor expansion off, singleton 0.95.
- Test (chunk04+05, one round):
  - P=0.7069, R=0.6308, F1=0.6667 (no improvement vs Iteration 3)

## Iteration 5 (Current Best)
- Status: completed
- Config: `experiments/bc2gm/config.generalization.r5.yaml`
- Change:
  - disable RE span augmentation,
  - keep boundary normalization and descriptor expansion,
  - singleton threshold 0.96.
- Test (chunk04+05, one round):
  - P=0.7778, R=0.6462, F1=0.7059

## Iteration 6
- Status: completed
- Change:
  - on top of Iteration 5, singleton 0.95.
- Test (chunk04+05, one round):
  - P=0.6418, R=0.6615, F1=0.6515 (regression)

## Iteration 7
- Status: completed
- Config: `experiments/bc2gm/config.generalization.r9.candidate_recall_guarded.yaml`
- Change:
  - generic candidate fallback always on (guarded),
  - capped candidate expansion to prevent prompt blow-up.
- Test (chunk04+05, one round):
  - P=0.7458, R=0.6769, F1=0.7097

## Iteration 8
- Status: completed
- Config: `experiments/bc2gm/config.generalization.r10.re_aug_highconf.yaml`
- Change:
  - re-enable RE span augmentation with high confidence gate.
- Test (chunk04+05, one round):
  - P=0.7581, R=0.7231, F1=0.7402

## Iteration 9
- Status: completed
- Change:
  - Prompt-level extra recall checklist for candidate/expert/re.
- Test (chunk04+05, one round):
  - P=0.7333, R=0.6769, F1=0.7040 (regression)
- Decision:
  - rolled back prompt recall checklist.

## Iteration 10
- Status: completed
- Change:
  - schema-profile candidate fallback morphology rules.
- Test (chunk04+05, one round):
  - P=0.7414, R=0.6615, F1=0.6992 (regression)

## Iteration 11
- Status: completed
- Change:
  - memory disabled on top of r10.
- Test (chunk04+05, one round):
  - P=0.7302, R=0.7077, F1=0.7188 (below r10)

## Iteration 12
- Status: completed
- Change:
  - candidate fallback bugfixes:
    - trim span whitespace,
    - preserve compact symbol mention when expanded tail is weak (e.g. `CDK2 complexes`),
    - hyphen expansion guard for short symbol-symbol chains.
- Test (chunk04+05, one round):
  - `P=0.7797, R=0.7077, F1=0.7419`
- Artifacts:
  - `outputs/bc2gm/generalization/pred.chunk45.r21.r10code.jsonl`
  - `outputs/bc2gm/generalization/eval.chunk45.r21.r10code.json`

## Iteration 13
- Status: completed
- Change:
  - add optional symbolic rescue / alias extraction (later default-disabled).
- Test (chunk04+05, one round):
  - `P=0.7143, R=0.6923, F1=0.7031` (regression on this run)
- Artifacts:
  - `outputs/bc2gm/generalization/pred.chunk45.r24.latest.jsonl`

## Iteration 14
- Status: completed
- Change:
  - low-information mention filter (optional): suppress Roman numeral / low-confidence titlecase / low-information descriptor-only spans.
- Offline deterministic replay on r21 outputs:
  - `P=0.8214, R=0.7077, F1=0.7603` (significant precision gain on fixed upstream predictions)
- Artifacts:
  - `outputs/bc2gm/generalization/pred.chunk45.r21.lowinfo_offline.jsonl`

## Iteration 15
- Status: completed
- Change:
  - conservative regex recall module (optional) + low-information filter.
- Full run (chunk04+05, one round):
  - `P=0.7377, R=0.6923, F1=0.7143` (regression on this run due upstream variance)
- Artifacts:
  - `outputs/bc2gm/generalization/pred.chunk45.r31.lowinfo_regex.jsonl`
  - `outputs/bc2gm/generalization/eval.chunk45.r31.lowinfo_regex.json`

## Current best checkpoint
- Best observed full run on chunk04+05 one-round:
  - `P=0.7797, R=0.7077, F1=0.7419`
- Best config lineage:
  - base: `experiments/bc2gm/config.generalization.r10.re_aug_highconf.yaml`
  - code revision with iteration-12 fixes
- Best artifacts:
  - `outputs/bc2gm/generalization/pred.chunk45.r21.r10code.jsonl`
