# Generalization Refactor Plan (Pause/Resume Safe)

## Objective
- Remove dataset-specific and content-specific heuristics from core runtime (`src/maner`).
- Keep the multi-agent pipeline reusable for arbitrary NER schemas/tasks.
- Preserve auditability and backward compatibility where practical.

## Scope
- In scope:
  - Candidate fallback logic
  - Pipeline postprocess/injection/filter logic
  - Config defaults and compatibility aliases
  - Tests and smoke validation
- Out of scope:
  - Rewriting experiment-only scripts under `experiments/bc2gm/`
  - Data preparation changes

## Step 1: Replace hardcoded domain fallback
- File: `src/maner/agents/candidate_agent.py`
- Action:
  - Remove biomedical fallback span injection.
  - Add generic fallback mode (config-driven, default safe).
- Checkpoint:
  - Candidate tests still pass.

## Step 2: De-specialize pipeline postprocess
- File: `src/maner/orchestrator/pipeline.py`
- Action:
  - Remove hardcoded high-value term injection and hardcoded blocklist.
  - Replace with config-driven lexical injection and blocklist.
  - Add generic descriptor expansion controls (off by default).
  - Replace gene-specific defaults with neutral defaults.
- Checkpoint:
  - Pipeline optimization/smoke tests pass.

## Step 3: Backward-compatible config aliasing
- Files:
  - `src/maner/orchestrator/pipeline.py`
  - `configs/default.yaml`
  - `configs/deepseek.yaml`
- Action:
  - Add aliases: `augmentation_entity_like_only` / `adjudicator_singleton_require_entity_like`.
  - Keep old keys functional for compatibility.
- Checkpoint:
  - Existing configs still runnable.

## Step 4: Update tests and run validation
- Files:
  - `tests/test_candidate_agent.py`
  - optional additions in pipeline tests
- Action:
  - Remove tests bound to biomedical hardcoded fallback.
  - Add tests for generic fallback and config-driven lexical filter/injection.
- Validation:
  - `pytest -q`
  - `python -m maner.cli.run_pipeline --config configs/default.yaml`

## Progress log
- Completed:
  - Step 1
  - Step 2
  - Step 3
  - Step 4
- Validation:
  - `python -m pytest -q` -> 43 passed
  - `python -m maner.cli.run_pipeline --config configs/default.yaml` -> success
  - `python -m maner.cli.run_eval --gold_path tests/fixtures/tiny_gold.jsonl --pred_path outputs/predictions.jsonl --schema_path tests/fixtures/schema_example.json` -> success

## Resume marker
- Current phase: Completed (ready for unbiased holdout experiments)
