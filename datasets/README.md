# Datasets Directory

Each dataset should live under `datasets/<dataset_name>/`.

## Required Files

- `schema.<dataset_name>.json`
- At least one split file such as:
  - `<dataset_name>_train.jsonl`
  - `<dataset_name>_dev.jsonl`
  - `<dataset_name>_test.jsonl`

Minimal JSONL line format:

```json
{"id":"sample-1","text":"John works at Acme.","gold_mentions":[{"start":0,"end":4,"ent_type":"PER","text":"John"}]}
```

Minimal schema format:

```json
{
  "dataset_name": "my_dataset",
  "entity_types": [
    {"name": "PER", "description": "Named individual person."},
    {"name": "ORG", "description": "Named organization or institution."}
  ],
  "relation_constraints": []
}
```

## Optional Files

- `dataset.source.yaml`
  - Optional bootstrap manifest for downloading or preparing the dataset when the JSONL files are missing.

Example `dataset.source.yaml`:

```yaml
commands:
  - python scripts/download_my_dataset.py
downloads:
  - url: https://example.com/my_dataset_test.jsonl
    output_path: datasets/my_dataset/my_dataset_test.jsonl
huggingface:
  dataset_name: conll2003
  format: token_classification
  split_map:
    train: train
    dev: validation
    test: test
  output_files:
    train: datasets/my_dataset/my_dataset_train.jsonl
    dev: datasets/my_dataset/my_dataset_dev.jsonl
    test: datasets/my_dataset/my_dataset_test.jsonl
  tokens_field: tokens
  tags_field: ner_tags
  sample_id_field: id
  label_list: ["O", "B-PER", "I-PER", "B-ORG", "I-ORG"]
```

## Main Entrypoint

Initialize a dataset workspace:

```bash
python experiments/run_dataset_eval.py init-dataset --dataset-id my_dataset
```

Validate the dataset:

```bash
python experiments/run_dataset_eval.py validate-dataset --dataset-id my_dataset --require-train
```

Bootstrap missing data:

```bash
python experiments/run_dataset_eval.py bootstrap-dataset --dataset-id my_dataset --include-train
```

Run full checkpoint evaluation:

```bash
python experiments/run_dataset_eval.py start --dataset-id my_dataset --background
```

Run chunked sample evaluation:

```bash
python experiments/run_dataset_eval.py chunked-eval --dataset-id my_dataset --max-samples 100
```

Generate the dataset-specific expert prompt overlay:

```bash
python experiments/run_dataset_eval.py generate-prompt --dataset-id my_dataset --force
```

Notes:

- The expert prompt generator is now `zero-shot`: it uses only `schema.<dataset_name>.json`, specifically the dataset name, entity type names, and entity type descriptions.
- No training split is required to generate the dataset-specific expert overlay.
- Evaluation requires the target split file, typically `<dataset_name>_test.jsonl`.
