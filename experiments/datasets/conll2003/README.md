# CoNLL2003 Evaluation Layout

## Structure

- `configs/`: all CoNLL2003 pipeline configs
- `prompts/`: CoNLL2003 prompt templates
- `results/`: evaluation outputs and checkpoint reports
- `logs/`: protocol log and console log
- `runtime/`: runtime metadata (`runtime.json`)
- `dataset.eval.yaml`: dataset profile for the unified entrypoint

## Unified Entrypoint

Use:

```bash
python experiments/run_dataset_eval.py <command> --profile experiments/datasets/conll2003/dataset.eval.yaml
```

Commands:

- `start`: start evaluation (`--background` for detached mode)
- `status`: show checkpoint progress and latest accuracy
- `pause`: pause current run
- `resume`: resume paused run
- `logs`: print console logs (`--follow` for real-time tail)

Examples:

```bash
python experiments/run_dataset_eval.py start --profile experiments/datasets/conll2003/dataset.eval.yaml --background
python experiments/run_dataset_eval.py status --profile experiments/datasets/conll2003/dataset.eval.yaml
python experiments/run_dataset_eval.py logs --profile experiments/datasets/conll2003/dataset.eval.yaml --follow
python experiments/run_dataset_eval.py pause --profile experiments/datasets/conll2003/dataset.eval.yaml
python experiments/run_dataset_eval.py resume --profile experiments/datasets/conll2003/dataset.eval.yaml
```
