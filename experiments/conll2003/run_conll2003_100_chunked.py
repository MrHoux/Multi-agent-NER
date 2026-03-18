from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from maner.core.schema import load_schema
from maner.eval.metrics import evaluate_from_files

from prepare_conll2003_data import prepare_conll2003


def _run_cmd(cmd: list[str], cwd: Path) -> None:
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(cwd))


def _append_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"- {ts}: {message}\n")


def _merge_jsonl_files(paths: list[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for p in paths:
            with p.open("r", encoding="utf-8") as src:
                for line in src:
                    if line.strip():
                        out.write(line)


def _run_pipeline_once(
    repo_root: Path,
    config_path: Path,
    data_path: Path,
    schema_path: Path,
    pred_path: Path,
) -> None:
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
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
    _run_cmd(cmd, cwd=repo_root)


def _eval_file(gold_path: Path, pred_path: Path, schema_path: Path) -> dict[str, Any]:
    schema = load_schema(schema_path)
    return evaluate_from_files(str(gold_path), str(pred_path), schema)


def _chunk_name(idx: int) -> str:
    return f"chunk{idx + 1:02d}"


def _evaluate_config(
    repo_root: Path,
    config_path: Path,
    schema_path: Path,
    merged_gold_path: Path,
    chunk_paths: list[Path],
    gate_chunk_count: int,
    output_root: Path,
    log_path: Path,
    threshold: float,
    drop_tolerance: float,
) -> tuple[bool, dict[str, Any]]:
    cfg_stem = config_path.stem
    cfg_out = output_root / cfg_stem
    cfg_out.mkdir(parents=True, exist_ok=True)

    first_two_scores: list[float] = []
    first_two_metrics: dict[str, Any] = {}
    for idx in range(gate_chunk_count):
        chunk = chunk_paths[idx]
        pred_path = cfg_out / f"pred.{_chunk_name(idx)}.jsonl"
        _run_pipeline_once(
            repo_root=repo_root,
            config_path=config_path,
            data_path=chunk,
            schema_path=schema_path,
            pred_path=pred_path,
        )
        metrics = _eval_file(gold_path=chunk, pred_path=pred_path, schema_path=schema_path)
        f1 = float(metrics["micro"]["f1"])
        first_two_scores.append(f1)
        first_two_metrics[_chunk_name(idx)] = metrics
        _append_log(
            log_path,
            f"{cfg_stem} {_chunk_name(idx)} micro_f1={f1:.4f}",
        )

    first_two_avg = sum(first_two_scores) / max(1, len(first_two_scores))
    _append_log(log_path, f"{cfg_stem} first_two_avg_f1={first_two_avg:.4f}")
    if first_two_avg < threshold:
        return False, {
            "config": str(config_path),
            "status": "fail_threshold",
            "first_two_avg_f1": first_two_avg,
            "first_two_metrics": first_two_metrics,
        }

    chunk_metrics: dict[str, Any] = {}
    pred_paths: list[Path] = []
    all_chunk_scores: list[float] = []
    for idx, chunk in enumerate(chunk_paths):
        pred_path = cfg_out / f"pred.{_chunk_name(idx)}.jsonl"
        if idx >= 2:
            _run_pipeline_once(
                repo_root=repo_root,
                config_path=config_path,
                data_path=chunk,
                schema_path=schema_path,
                pred_path=pred_path,
            )
        metrics = _eval_file(gold_path=chunk, pred_path=pred_path, schema_path=schema_path)
        f1 = float(metrics["micro"]["f1"])
        pred_paths.append(pred_path)
        chunk_metrics[_chunk_name(idx)] = metrics
        all_chunk_scores.append(f1)
        _append_log(log_path, f"{cfg_stem} {_chunk_name(idx)} micro_f1={f1:.4f}")

    merged_pred = cfg_out / "pred.full100.jsonl"
    _merge_jsonl_files(pred_paths, merged_pred)
    full_metrics = _eval_file(
        gold_path=merged_gold_path,
        pred_path=merged_pred,
        schema_path=schema_path,
    )
    full_f1 = float(full_metrics["micro"]["f1"])
    _append_log(log_path, f"{cfg_stem} full100 micro_f1={full_f1:.4f}")

    if full_f1 + drop_tolerance < first_two_avg:
        _append_log(
            log_path,
            (
                f"{cfg_stem} rejected for generalization_drop: "
                f"first_two_avg={first_two_avg:.4f}, full100={full_f1:.4f}, "
                f"drop_tolerance={drop_tolerance:.4f}"
            ),
        )
        return False, {
            "config": str(config_path),
            "status": "fail_generalization_drop",
            "first_two_avg_f1": first_two_avg,
            "full100_f1": full_f1,
            "chunk_metrics": chunk_metrics,
            "full_metrics": full_metrics,
        }

    return True, {
        "config": str(config_path),
        "status": "selected",
        "first_two_avg_f1": first_two_avg,
        "full100_f1": full_f1,
        "chunk_metrics": chunk_metrics,
        "full_metrics": full_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run CoNLL2003 100-sample evaluation in 5 chunks, with automatic optimization "
            "config fallback until threshold and generalization checks pass."
        )
    )
    parser.add_argument("--dataset_name", default="conll2003")
    parser.add_argument("--split", default="test")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--chunk_size", type=int, default=20)
    parser.add_argument(
        "--gate_samples",
        type=int,
        default=40,
        help="Number of leading samples used as a quick optimization gate.",
    )
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--drop_tolerance", type=float, default=0.08)
    parser.add_argument(
        "--config_candidates",
        nargs="+",
        default=[
            "experiments/datasets/conll2003/configs/config.conll2003.base.yaml",
            "experiments/datasets/conll2003/configs/config.conll2003.optim.recall.yaml",
            "experiments/datasets/conll2003/configs/config.conll2003.optim.recall_disamb.yaml",
            "experiments/datasets/conll2003/configs/config.conll2003.optim.disamb.yaml",
            "experiments/datasets/conll2003/configs/config.conll2003.optim.balance.v2.yaml",
            "experiments/datasets/conll2003/configs/config.conll2003.optim.recall_plus.yaml",
        ],
    )
    parser.add_argument(
        "--log_path",
        default="experiments/datasets/conll2003/logs/EXECUTION_PROTOCOL_AND_LOG.md",
    )
    parser.add_argument(
        "--report_path",
        default="",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    log_path = repo_root / args.log_path
    range_suffix = f"{args.start_index}_{args.start_index + args.max_samples - 1}"
    report_rel = (
        args.report_path
        if args.report_path.strip()
        else f"experiments/datasets/conll2003/results/chunks_{range_suffix}/selection_report.json"
    )
    report_path = repo_root / report_rel
    output_root = report_path.parent
    output_root.mkdir(parents=True, exist_ok=True)

    gate_chunk_count = max(1, min(args.max_samples // args.chunk_size, args.gate_samples // args.chunk_size))
    if gate_chunk_count <= 0:
        gate_chunk_count = 1

    _append_log(
        log_path,
        (
            "start optimization cycle: "
            f"threshold={args.threshold:.2f}, drop_tolerance={args.drop_tolerance:.2f}, "
            f"start_index={args.start_index}, max_samples={args.max_samples}, "
            f"chunk_size={args.chunk_size}, gate_samples={args.gate_samples}, "
            f"gate_chunk_count={gate_chunk_count}"
        ),
    )

    data_path = repo_root / "datasets" / "conll2003" / f"conll2003_{args.split}.{range_suffix}.jsonl"
    schema_path = repo_root / "datasets" / "conll2003" / "schema.conll2003.json"
    chunks_dir = repo_root / "datasets" / "conll2003" / f"chunks_{range_suffix}"

    _, _, chunk_paths = prepare_conll2003(
        dataset_name=args.dataset_name,
        split=args.split,
        start_index=args.start_index,
        max_samples=args.max_samples,
        chunk_size=args.chunk_size,
        output_jsonl=data_path,
        output_schema=schema_path,
        output_chunks_dir=chunks_dir,
    )
    chunk_paths = sorted(chunk_paths)[:5]
    if len(chunk_paths) < 5:
        raise RuntimeError(f"Expected 5 chunks, got {len(chunk_paths)}")

    all_results: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None

    for cfg in args.config_candidates:
        cfg_path = repo_root / cfg
        if not cfg_path.exists():
            _append_log(log_path, f"skip missing config: {cfg_path}")
            continue
        _append_log(log_path, f"evaluate config: {cfg_path}")
        ok, result = _evaluate_config(
            repo_root=repo_root,
            config_path=cfg_path,
            schema_path=schema_path,
            merged_gold_path=data_path,
            chunk_paths=chunk_paths,
            gate_chunk_count=gate_chunk_count,
            output_root=output_root,
            log_path=log_path,
            threshold=args.threshold,
            drop_tolerance=args.drop_tolerance,
        )
        all_results.append(result)
        if ok:
            selected = result
            _append_log(log_path, f"selected config: {cfg_path}")
            break
        _append_log(log_path, f"config rejected: {cfg_path} status={result.get('status')}")

    summary = {
        "threshold": args.threshold,
        "drop_tolerance": args.drop_tolerance,
        "start_index": args.start_index,
        "max_samples": args.max_samples,
        "chunk_size": args.chunk_size,
        "gate_samples": args.gate_samples,
        "gate_chunk_count": gate_chunk_count,
        "data_path": str(data_path),
        "results": all_results,
        "selected": selected,
    }
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if selected is None:
        _append_log(log_path, "no config satisfied thresholds; optimization loop requires further changes")
        raise SystemExit("No configuration satisfied threshold/generalization checks.")

    print(json.dumps(selected, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
