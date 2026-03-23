# CoNLL2003 Experiment

This folder adapts the project from BC2GM-focused runs to CoNLL2003 newswire NER.

## Files
- `prepare_conll2003_data.py`:
  - Loads HuggingFace `conll2003`.
  - Converts BIO tags to span-level JSONL (`id`, `text`, `gold_mentions`).
  - Writes CoNLL2003 schema with `PER/ORG/LOC/MISC`.
  - Splits 100 samples into 5 chunks (`20 x 5`).
- `configs/prompts_cot.yaml`:
  - Shared generic prompt set used by all non-expert agents.
- `configs/runtime.deepseek.all_agents.yaml`:
  - Shared dataset-agnostic runtime config used by the generic runners.
- `experiments/datasets/conll2003/prompts/expert.generated.yaml`:
  - Dataset-specific `expert_agent` overlay generated from a training slice.
- `experiments/datasets/conll2003/prompts/expert.generated.meta.json`:
  - Prompt design summary and generation metadata.
- `experiments/datasets/conll2003/results/`:
  - All CoNLL2003 evaluation outputs.
- `experiments/datasets/conll2003/logs/`:
  - Console/protocol logs.
- `experiments/datasets/conll2003/dataset.eval.yaml`:
  - Auto-generated profile used by the generic dataset entrypoint.
- `experiments/run_dataset_100_chunked.py`:
  - Generic chunked evaluation runner for any dataset in standard JSONL format.
- `experiments/run_dataset_full_checkpoints.py`:
  - Generic full-test checkpoint runner for any dataset in standard JSONL format.
- `EXECUTION_PROTOCOL_AND_LOG.md`:
  - Behavior protocol and action trace log.

## Run (WSL NER environment)
Unified entrypoint:
```bash
python experiments/run_dataset_eval.py start \
  --dataset-id conll2003 \
  --background
```

Check accuracy/progress at any time:
```bash
python experiments/run_dataset_eval.py status \
  --dataset-id conll2003
```

Real-time console log:
```bash
python experiments/run_dataset_eval.py logs \
  --dataset-id conll2003 \
  --follow
```

Pause/Resume:
```bash
python experiments/run_dataset_eval.py pause  --dataset-id conll2003
python experiments/run_dataset_eval.py resume --dataset-id conll2003
```

Windowed chunked evaluation:
```bash
python experiments/run_dataset_eval.py chunked-eval \
  --dataset-id conll2003 \
  --start-index 100 \
  --max_samples 100 \
  --gate_samples 40 \
  --threshold 0.75
```

Run full test with 100-sample checkpoints, optimize on failing node:
```bash
python experiments/run_dataset_eval.py start \
  --dataset-id conll2003 \
  --background
```

Results:
- Selection report: `experiments/datasets/conll2003/results/chunks_<start>_<end>/selection_report.json`
- Full-test checkpoints/report: `experiments/datasets/conll2003/results/full_test/`
