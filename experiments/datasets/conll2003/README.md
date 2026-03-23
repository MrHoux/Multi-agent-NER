# CoNLL2003 Evaluation Layout

## Structure

- `configs/`: reserved dataset config folder; the active chat runtime config is shared at `configs/runtime.deepseek.all_agents.yaml`
- `prompts/`: generated dataset-specific expert prompt overlays and prompt metadata
- `results/`: evaluation outputs and checkpoint reports
- `logs/`: protocol log and console log
- `runtime/`: runtime metadata (`runtime.json`)
- `dataset.eval.yaml`: dataset profile for the unified entrypoint

## Prompt Layout

- Shared base prompts for all non-expert roles live in `configs/prompts_cot.yaml`.
- Each dataset stores only its generated expert-role overlays in `prompts/`.
- The runtime merges the shared base prompt file with the dataset overlay at startup.

## Unified Entrypoint

Use:

```bash
python experiments/run_dataset_eval.py <command> --dataset-id conll2003
```

The legacy scripts under `experiments/conll2003/` are compatibility wrappers only. The unified entrypoint above is the authoritative path.

Commands:

- `start`: start evaluation (`--background` for detached mode)
- `chunked-eval`: run windowed chunk evaluation
- `status`: show checkpoint progress and latest accuracy
- `pause`: pause current run
- `resume`: resume paused run
- `logs`: print console logs (`--follow` for real-time tail)
- `generate-prompt`: regenerate the dataset-specific expert overlay
- `init-dataset`: scaffold the standard dataset layout

Examples:

```bash
python experiments/run_dataset_eval.py start --dataset-id conll2003 --background
python experiments/run_dataset_eval.py chunked-eval --dataset-id conll2003 --max-samples 100
python experiments/run_dataset_eval.py status --dataset-id conll2003
python experiments/run_dataset_eval.py logs --dataset-id conll2003 --follow
python experiments/run_dataset_eval.py pause --dataset-id conll2003
python experiments/run_dataset_eval.py resume --dataset-id conll2003
```
