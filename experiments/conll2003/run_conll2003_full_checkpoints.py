from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _repo_rel(path: Path) -> str:
    return str(path.resolve().relative_to(_repo_root().resolve())).replace("\\", "/")


def _discover_split_path(dataset_id: str, split: str) -> Path:
    dataset_dir = _repo_root() / "datasets" / dataset_id
    exact = dataset_dir / f"{dataset_id}_{split}.jsonl"
    if exact.exists():
        return exact.resolve()
    candidates = sorted(dataset_dir.glob(f"{dataset_id}_{split}*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"Missing split file for dataset={dataset_id} split={split}")
    return candidates[0].resolve()


def _prompt_overlay_path(dataset_id: str) -> Path | None:
    path = _repo_root() / "experiments" / "datasets" / dataset_id / "prompts" / "expert.generated.yaml"
    if path.exists():
        return path.resolve()
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compatibility wrapper. For new usage, prefer "
            "`python experiments/run_dataset_eval.py start --dataset-id conll2003`."
        )
    )
    parser.add_argument("--dataset_name", default="conll2003")
    parser.add_argument("--split", default="test")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=3453)
    parser.add_argument("--checkpoint_size", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--active_config", default="configs/runtime.deepseek.all_agents.yaml")
    parser.add_argument(
        "--optimization_configs",
        nargs="+",
        default=["configs/runtime.deepseek.all_agents.yaml"],
    )
    parser.add_argument(
        "--log_path",
        default="experiments/datasets/conll2003/logs/EXECUTION_PROTOCOL_AND_LOG.md",
    )
    parser.add_argument(
        "--report_path",
        default="experiments/datasets/conll2003/results/full_test/checkpoint_report.json",
    )
    parser.add_argument(
        "--progress_path",
        default="experiments/datasets/conll2003/results/full_test/checkpoint_progress.json",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_nodes", type=int, default=0)
    args = parser.parse_args()

    repo_root = _repo_root()
    schema_path = repo_root / "datasets" / args.dataset_name / f"schema.{args.dataset_name}.json"
    source_data_path = _discover_split_path(args.dataset_name, args.split)
    prompt_overlay_path = _prompt_overlay_path(args.dataset_name)

    command = [
        sys.executable,
        str(repo_root / "experiments" / "run_dataset_full_checkpoints.py"),
        "--dataset_name",
        args.dataset_name,
        "--split",
        args.split,
        "--source_data_path",
        _repo_rel(source_data_path),
        "--schema_path",
        _repo_rel(schema_path),
        "--reader_type",
        "generic_jsonl",
        "--start_index",
        str(args.start_index),
        "--max_samples",
        str(args.max_samples),
        "--checkpoint_size",
        str(args.checkpoint_size),
        "--threshold",
        str(args.threshold),
        "--active_config",
        args.active_config,
        "--optimization_configs",
        *args.optimization_configs,
        "--log_path",
        args.log_path,
        "--report_path",
        args.report_path,
        "--progress_path",
        args.progress_path,
    ]
    if prompt_overlay_path is not None:
        command.extend(["--prompt_overlay_path", _repo_rel(prompt_overlay_path)])
    if args.resume:
        command.append("--resume")
    if args.max_nodes > 0:
        command.extend(["--max_nodes", str(args.max_nodes)])

    raise SystemExit(subprocess.run(command, cwd=str(repo_root), check=False).returncode)


if __name__ == "__main__":
    main()
