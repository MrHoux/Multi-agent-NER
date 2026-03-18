from __future__ import annotations

import argparse
from pathlib import Path

from maner.core.schema import load_schema
from maner.eval.metrics import evaluate_from_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate existing per-chunk predictions for CoNLL2003 full-test checkpoints."
    )
    parser.add_argument(
        "--repo_root",
        default=".",
        help="Project root (default: current directory).",
    )
    parser.add_argument(
        "--chunks_dir",
        default="datasets/conll2003/chunks_0_3452_100",
    )
    parser.add_argument(
        "--schema_path",
        default="datasets/conll2003/schema.conll2003.json",
    )
    parser.add_argument(
        "--output_root",
        default="outputs/conll2003/full_test_3453",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    chunks_dir = repo_root / args.chunks_dir
    schema_path = repo_root / args.schema_path
    output_root = repo_root / args.output_root

    schema = load_schema(schema_path)

    candidates: dict[str, list[str]] = {
        "config.conll2003.optim.recall_disamb": [
            "pred.chunk01.jsonl",
            "pred.chunk02.jsonl",
            "pred.chunk03.jsonl",
            "pred.chunk04.jsonl",
            "pred.chunk05.jsonl",
            "pred.chunk05.rerun2.jsonl",
        ],
        "config.conll2003.optim.recall": [
            "pred.chunk04.jsonl",
            "pred.chunk05.jsonl",
        ],
        "config.conll2003.optim.disamb": [
            "pred.chunk04.jsonl",
            "pred.chunk05.jsonl",
        ],
        "config.conll2003.optim.recall_disamb.r2": [
            "pred.chunk05.jsonl",
        ],
    }

    for cfg_name, pred_files in candidates.items():
        cfg_dir = output_root / cfg_name
        if not cfg_dir.exists():
            continue
        print(f"== {cfg_name} ==")
        for pred_name in pred_files:
            pred_path = cfg_dir / pred_name
            if not pred_path.exists():
                continue
            chunk_name = pred_name.replace("pred.", "").replace(".rerun2", "")
            gold_path = chunks_dir / f"conll2003_test.{chunk_name}"
            metrics = evaluate_from_files(str(gold_path), str(pred_path), schema)
            f1 = float(metrics["micro"]["f1"])
            status = "PASS" if f1 >= args.threshold else "FAIL"
            print(f"{pred_name}: f1={f1:.4f} [{status}]")
        print()


if __name__ == "__main__":
    main()
