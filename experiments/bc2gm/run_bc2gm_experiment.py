from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from prepare_bc2gm_data import prepare_bc2gm


def _run_cmd(cmd: list[str]) -> None:
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare BC2GM from HuggingFace and run MANER pipeline+eval"
    )
    parser.add_argument("--dataset_name", default="spyysalo/bc2gm_corpus")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument(
        "--config",
        default="experiments/bc2gm/config.bc2gm.deepseek.yaml",
        help="Unified experiment config path",
    )
    parser.add_argument(
        "--pred_path",
        default="outputs/bc2gm/predictions.deepseek.real.jsonl",
        help="Prediction output path",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    data_path = repo_root / "datasets" / "bc2gm" / "bc2gm_test.jsonl"
    schema_path = repo_root / "datasets" / "bc2gm" / "schema.bc2gm.json"
    config_path = repo_root / args.config
    pred_path = repo_root / args.pred_path

    prepare_bc2gm(
        dataset_name=args.dataset_name,
        split=args.split,
        max_samples=args.max_samples,
        output_jsonl=data_path,
        output_schema=schema_path,
    )

    _run_cmd(
        [
            sys.executable,
            "-m",
            "maner.cli.run_pipeline",
            "--config",
            str(config_path),
            "--set",
            f"data.data_path={data_path}",
            "--set",
            f"data.schema_path={schema_path}",
            "--set",
            f"output.predictions_path={pred_path}",
        ]
    )

    _run_cmd(
        [
            sys.executable,
            "-m",
            "maner.cli.run_eval",
            "--gold_path",
            str(data_path),
            "--pred_path",
            str(pred_path),
            "--schema_path",
            str(schema_path),
        ]
    )


if __name__ == "__main__":
    main()
