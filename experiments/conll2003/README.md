# CoNLL2003 Experiment

This folder adapts the project from BC2GM-focused runs to CoNLL2003 newswire NER.

## Files
- `prepare_conll2003_data.py`:
  - Loads HuggingFace `conll2003`.
  - Converts BIO tags to span-level JSONL (`id`, `text`, `gold_mentions`).
  - Writes CoNLL2003 schema with `PER/ORG/LOC/MISC`.
  - Splits 100 samples into 5 chunks (`20 x 5`).
- `experiments/datasets/conll2003/prompts/prompts.conll2003.news.yaml`:
  - Prompt set with explicit CoNLL2003 label policy.
- `experiments/datasets/conll2003/configs/`:
  - All CoNLL2003 configs are centralized here.
- `experiments/datasets/conll2003/results/`:
  - All CoNLL2003 evaluation outputs.
- `experiments/datasets/conll2003/logs/`:
  - Console/protocol logs.
- `experiments/datasets/conll2003/dataset.eval.yaml`:
  - Unified profile used by the dataset entrypoint.
- `run_conll2003_100_chunked.py`:
  - Automated threshold loop:
    - first two chunks must average `>= 0.70`
    - then run all 5 chunks and reject if full-100 drops materially
    - otherwise select the config.
- `EXECUTION_PROTOCOL_AND_LOG.md`:
  - Behavior protocol and action trace log.

## Run (WSL NER environment)
Unified entrypoint:
```bash
python experiments/run_dataset_eval.py start \
  --profile experiments/datasets/conll2003/dataset.eval.yaml \
  --background
```

Check accuracy/progress at any time:
```bash
python experiments/run_dataset_eval.py status \
  --profile experiments/datasets/conll2003/dataset.eval.yaml
```

Real-time console log:
```bash
python experiments/run_dataset_eval.py logs \
  --profile experiments/datasets/conll2003/dataset.eval.yaml \
  --follow
```

Pause/Resume:
```bash
python experiments/run_dataset_eval.py pause  --profile experiments/datasets/conll2003/dataset.eval.yaml
python experiments/run_dataset_eval.py resume --profile experiments/datasets/conll2003/dataset.eval.yaml
```

Legacy direct scripts:
```bash
python experiments/conll2003/run_conll2003_100_chunked.py
```

Run a specific 100-sample window, e.g. `100-199`, with first-40 gate target `>0.75`:
```bash
python experiments/conll2003/run_conll2003_100_chunked.py \
  --start_index 100 \
  --max_samples 100 \
  --gate_samples 40 \
  --threshold 0.75
```

Run full test with 100-sample checkpoints, optimize on failing node:
```bash
python experiments/conll2003/run_conll2003_full_checkpoints.py \
  --start_index 0 \
  --max_samples 3453 \
  --checkpoint_size 100 \
  --threshold 0.75
```

Results:
- Selection report: `experiments/datasets/conll2003/results/chunks_<start>_<end>/selection_report.json`
- Full-test checkpoints/report: `experiments/datasets/conll2003/results/full_test_all_agents/`
