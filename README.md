# multiagent-ner

Training-free multi-agent NER pipeline with:

- `direct NER` as the primary extraction line
- `candidate` as span supplementation
- `expert` for schema-grounded domain reasoning
- `in-context` for sentence-level contextual reasoning
- `RAG` for optional Wikipedia-backed disambiguation
- `adjudicator` for final semantic arbitration
- `verifier` for structural validation only

The repository is packaged so another machine can clone it, add a dataset in the expected format, set API credentials, and run.

## 1. Install

```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1
python -m pip install -e .
```

## 2. Required Configuration

Main config files:

- `configs/default.yaml`
- `configs/deepseek.yaml`
- `configs/prompts_cot.yaml`
- `configs/runtime.deepseek.all_agents.yaml`

Dataset-specific runtime files for the current CoNLL2003 setup:

- `datasets/conll2003/schema.conll2003.json`
- `experiments/datasets/conll2003/dataset.eval.yaml`
- `experiments/datasets/conll2003/prompts/expert.generated.yaml`

`logs/`, `results/`, and `runtime/` under `experiments/datasets/<dataset>/` are runtime directories. They are auto-created when needed and are ignored by git.

## 3. API Credentials

Set your provider credentials in:

- `configs/deepseek.yaml`

If you want environment-variable based configuration, use:

- `.env.example`

## 4. Dataset Format

Expected JSONL format:

```json
{"id":"s1","text":"sample text","gold_mentions":[{"start":0,"end":4,"ent_type":"PER"}]}
```

Required fields:

- `id`
- `text`

Optional for evaluation:

- `gold_mentions`

Schema format:

```json
{
  "dataset_name": "my_dataset",
  "entity_types": [
    {"name": "PER", "description": "Person entity."}
  ],
  "relation_constraints": []
}
```

## 5. Main Entrypoints

Low-level pipeline:

```bash
python -m maner.cli.run_pipeline --config configs/runtime.deepseek.all_agents.yaml
```

Dataset runner:

```bash
python experiments/run_dataset_eval.py start --dataset-id conll2003 --background
```

Status:

```bash
python experiments/run_dataset_eval.py status --dataset-id conll2003
```

Logs:

```bash
python experiments/run_dataset_eval.py logs --dataset-id conll2003 --follow
```

Evaluation:

```bash
python -m maner.cli.run_eval --gold_path <gold.jsonl> --pred_path <pred.jsonl> --schema_path <schema.json>
```

## 6. Logging

Pipeline logs are timestamped and emitted in a stable key-value format:

```text
[2026-03-25 00:51:35] [pipeline] sample_start | chars=55 idx=1 sample_id=test-00000
```

When `pipeline.progress_agent_trace=true`, agent-level timing is also logged:

```text
[2026-03-25 00:51:42] [pipeline] ner_direct_agent_done | elapsed_ms=7689 llm_calls=1 mentions=2 sample_id=test-00000
```

This format is designed for direct reading and for downstream parsing.

## 7. Runtime Outputs

Prediction files contain:

- `id`
- `text`
- `mentions`
- `traces`
- `costs`

The repository does not keep historical runtime artifacts by default. Temporary logs, PID files, sqlite memory files, and generated results are ignored by `.gitignore`.

## 8. Evaluation Metric

Primary metric is strict span-level exact-match micro-F1:

- `Precision = TP / (TP + FP)`
- `Recall = TP / (TP + FN)`
- `F1 = 2PR / (P + R)`

An entity counts as correct only when both:

- span offsets match exactly
- entity type matches exactly

## 9. Key Modules

- `src/maner/core/`
- `src/maner/llm/`
- `src/maner/agents/`
- `src/maner/orchestrator/`
- `src/maner/memory/`
- `src/maner/eval/`
- `experiments/`

## 10. Repository State

The repository is now kept in a handoff-safe state:

- source code retained
- required configs retained
- dataset definition retained
- prompt files retained
- runtime artifacts cleaned
- temporary caches ignored
