# BC2GM Experiment (Isolated Files)

This folder adds a HuggingFace BC2GM test workflow without modifying the core source files.

## What is added

- `prepare_bc2gm_data.py`:
  - Loads `spyysalo/bc2gm_corpus` via `from datasets import load_dataset`
  - Converts BIO token tags to span-level JSONL (`id`, `text`, `gold_mentions`)
  - Writes a schema file for this dataset
- `prompts.biomed_cot.yaml`:
  - Prompt set where `ExpertAgent` is explicitly instructed as a biomedical expert
- `config.bc2gm.deepseek.yaml`:
  - Unified BC2GM config (single source of truth for 5/20/N sample runs)
- `run_bc2gm_experiment.py`:
  - One-command prepare + pipeline + eval with runtime `--set` overrides

## Why import handling is special

This repository has a local folder named `datasets/`, which can shadow the HuggingFace `datasets` package.
`prepare_bc2gm_data.py` contains a safe import helper to force importing the real HuggingFace package.

## Quick start

```bash
python -m pip install datasets
python experiments/bc2gm/run_bc2gm_experiment.py --split test --max_samples 50
```

Use the same config for different sample sizes or outputs:

```bash
python -m maner.cli.run_pipeline \
  --config experiments/bc2gm/config.bc2gm.deepseek.yaml \
  --set data.data_path=outputs/bc2gm/bc2gm_test.20.jsonl \
  --set output.predictions_path=outputs/bc2gm/predictions.deepseek.real.20.jsonl
```

## Output paths

- Converted BC2GM gold data: `datasets/bc2gm/bc2gm_test.jsonl`
- BC2GM schema: `datasets/bc2gm/schema.bc2gm.json`
- Predictions: default `outputs/bc2gm/predictions.deepseek.real.jsonl` (overridable)
