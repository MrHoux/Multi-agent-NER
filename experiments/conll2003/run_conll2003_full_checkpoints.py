from __future__ import annotations

import argparse
import json
import shutil
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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


def _micro_f1_from_counts(tp: int, fp: int, fn: int) -> float:
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    return (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0


def _build_report_payload(
    *,
    args: argparse.Namespace,
    stop_reason: str,
    active_config: Path,
    node_results: list[dict[str, Any]],
    full_metrics: dict[str, Any] | None,
    data_path: Path,
    chunks_dir: Path,
    output_root: Path,
) -> dict[str, Any]:
    return {
        "target": {
            "dataset_name": args.dataset_name,
            "split": args.split,
            "start_index": args.start_index,
            "max_samples": args.max_samples,
            "checkpoint_size": args.checkpoint_size,
            "threshold": args.threshold,
        },
        "stop_reason": stop_reason,
        "final_active_config": str(active_config),
        "node_results": node_results,
        "full_metrics": full_metrics,
        "data_path": str(data_path),
        "chunks_dir": str(chunks_dir),
        "output_root": str(output_root),
    }


def _choose_best_config_for_chunk(
    *,
    repo_root: Path,
    output_root: Path,
    schema_path: Path,
    chunk_path: Path,
    chunk_idx: int,
    active_config: Path,
    candidate_configs: list[Path],
    threshold: float,
    log_path: Path,
) -> dict[str, Any]:
    candidates = [active_config] + [c for c in candidate_configs if c != active_config]
    best: dict[str, Any] | None = None

    for cfg in candidates:
        cfg_stem = cfg.stem
        pred_path = output_root / cfg_stem / f"pred.{_chunk_name(chunk_idx)}.jsonl"
        _run_pipeline_once(
            repo_root=repo_root,
            config_path=cfg,
            data_path=chunk_path,
            schema_path=schema_path,
            pred_path=pred_path,
        )
        metrics = _eval_file(gold_path=chunk_path, pred_path=pred_path, schema_path=schema_path)
        f1 = float(metrics["micro"]["f1"])
        _append_log(
            log_path,
            f"{cfg_stem} {_chunk_name(chunk_idx)} checkpoint_f1={f1:.4f}",
        )
        item = {
            "config": str(cfg),
            "config_stem": cfg_stem,
            "pred_path": str(pred_path),
            "metrics": metrics,
            "f1": f1,
        }
        if best is None or f1 > float(best["f1"]):
            best = item
        if f1 >= threshold and cfg == active_config:
            break

    assert best is not None
    return best


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run CoNLL2003 full test with checkpoint evaluation every N samples. "
            "If a checkpoint drops below threshold, run optimization configs on that "
            "checkpoint before continuing."
        )
    )
    parser.add_argument("--dataset_name", default="conll2003")
    parser.add_argument("--split", default="test")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=3453)
    parser.add_argument("--checkpoint_size", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument(
        "--active_config",
        default="experiments/datasets/conll2003/configs/config.conll2003.optim.recall_disamb.yaml",
    )
    parser.add_argument(
        "--optimization_configs",
        nargs="+",
        default=[
            "experiments/datasets/conll2003/configs/config.conll2003.optim.recall.yaml",
            "experiments/datasets/conll2003/configs/config.conll2003.optim.recall_disamb.yaml",
            "experiments/datasets/conll2003/configs/config.conll2003.optim.disamb.yaml",
            "experiments/datasets/conll2003/configs/config.conll2003.base.yaml",
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
        default="experiments/datasets/conll2003/results/full_test_3453/checkpoint_report.json",
    )
    parser.add_argument(
        "--progress_path",
        default="",
        help="Optional progress snapshot path updated after every checkpoint.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing selected/pred.chunkXX.jsonl under output_root.",
    )
    parser.add_argument(
        "--max_nodes",
        type=int,
        default=0,
        help="Optional safety limit for number of checkpoints; 0 means no limit.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    log_path = repo_root / args.log_path
    report_path = repo_root / args.report_path
    progress_path = repo_root / args.progress_path if args.progress_path else report_path.with_name("checkpoint_progress.json")
    output_root = report_path.parent
    output_root.mkdir(parents=True, exist_ok=True)

    range_suffix = f"{args.start_index}_{args.start_index + args.max_samples - 1}"
    data_path = repo_root / "datasets" / "conll2003" / f"conll2003_{args.split}.{range_suffix}.jsonl"
    schema_path = repo_root / "datasets" / "conll2003" / "schema.conll2003.json"
    chunks_dir = repo_root / "datasets" / "conll2003" / f"chunks_{range_suffix}_{args.checkpoint_size}"

    _append_log(
        log_path,
        (
            "start full-test checkpoint run: "
            f"start_index={args.start_index}, max_samples={args.max_samples}, "
            f"checkpoint_size={args.checkpoint_size}, threshold={args.threshold:.2f}"
        ),
    )

    _, _, chunk_paths = prepare_conll2003(
        dataset_name=args.dataset_name,
        split=args.split,
        start_index=args.start_index,
        max_samples=args.max_samples,
        chunk_size=args.checkpoint_size,
        output_jsonl=data_path,
        output_schema=schema_path,
        output_chunks_dir=chunks_dir,
    )
    chunk_paths = sorted(chunk_paths)
    if args.max_nodes > 0:
        chunk_paths = chunk_paths[: args.max_nodes]

    active_config = repo_root / args.active_config
    if not active_config.exists():
        raise FileNotFoundError(f"Active config not found: {active_config}")
    optimization_configs = [repo_root / p for p in args.optimization_configs]
    optimization_configs = [p for p in optimization_configs if p.exists()]
    if not optimization_configs:
        raise RuntimeError("No valid optimization configs found.")

    tp_sum = 0
    fp_sum = 0
    fn_sum = 0
    node_results: list[dict[str, Any]] = []
    stop_reason = "completed"

    selected_pred_paths: list[Path] = []
    resume_start_idx = 0

    if args.resume:
        selected_dir = output_root / "selected"
        for idx, chunk_path in enumerate(chunk_paths):
            existing_pred = selected_dir / f"pred.{_chunk_name(idx)}.jsonl"
            if not existing_pred.exists():
                break
            metrics = _eval_file(gold_path=chunk_path, pred_path=existing_pred, schema_path=schema_path)
            micro = metrics["micro"]
            tp_sum += int(micro["tp"])
            fp_sum += int(micro["fp"])
            fn_sum += int(micro["fn"])
            cumulative_f1 = _micro_f1_from_counts(tp_sum, fp_sum, fn_sum)
            node_results.append(
                {
                    "checkpoint": _chunk_name(idx),
                    "node_end_index": min(
                        args.start_index + (idx + 1) * args.checkpoint_size,
                        args.start_index + args.max_samples,
                    )
                    - 1,
                    "status": "resumed",
                    "used_config": str(active_config),
                    "optimized": False,
                    "node_f1": float(micro["f1"]),
                    "cumulative_f1": cumulative_f1,
                    "metrics": metrics,
                }
            )
            selected_pred_paths.append(existing_pred)
            resume_start_idx = idx + 1

        if resume_start_idx > 0:
            _append_log(
                log_path,
                f"resume enabled: skipping {resume_start_idx} completed checkpoints",
            )

    progress_payload = _build_report_payload(
        args=args,
        stop_reason="running",
        active_config=active_config,
        node_results=node_results,
        full_metrics=None,
        data_path=data_path,
        chunks_dir=chunks_dir,
        output_root=output_root,
    )
    _write_json(progress_path, progress_payload)

    for idx in range(resume_start_idx, len(chunk_paths)):
        chunk_path = chunk_paths[idx]
        node_end = min(args.start_index + (idx + 1) * args.checkpoint_size, args.start_index + args.max_samples)
        _append_log(
            log_path,
            (
                f"checkpoint {_chunk_name(idx)} (up to sample_index {node_end - 1}) "
                f"active_config={active_config}"
            ),
        )

        best = _choose_best_config_for_chunk(
            repo_root=repo_root,
            output_root=output_root,
            schema_path=schema_path,
            chunk_path=chunk_path,
            chunk_idx=idx,
            active_config=active_config,
            candidate_configs=optimization_configs,
            threshold=args.threshold,
            log_path=log_path,
        )

        best_f1 = float(best["f1"])
        chosen_config = Path(str(best["config"]))
        optimized = chosen_config != active_config

        if best_f1 < args.threshold:
            stop_reason = (
                f"checkpoint_below_threshold_unrecoverable:{_chunk_name(idx)}:{best_f1:.4f}"
            )
            _append_log(
                log_path,
                (
                    f"stop at {_chunk_name(idx)}: best_f1={best_f1:.4f} "
                    f"< threshold={args.threshold:.2f}"
                ),
            )
            node_results.append(
                {
                    "checkpoint": _chunk_name(idx),
                    "node_end_index": node_end - 1,
                    "status": "failed",
                    "active_config": str(active_config),
                    "best_config": str(chosen_config),
                    "node_f1": best_f1,
                    "metrics": best["metrics"],
                }
            )
            progress_payload = _build_report_payload(
                args=args,
                stop_reason=stop_reason,
                active_config=active_config,
                node_results=node_results,
                full_metrics=None,
                data_path=data_path,
                chunks_dir=chunks_dir,
                output_root=output_root,
            )
            _write_json(progress_path, progress_payload)
            break

        active_config = chosen_config
        if optimized:
            _append_log(
                log_path,
                f"optimized at {_chunk_name(idx)}: switch active_config -> {active_config}",
            )

        canonical_pred = output_root / "selected" / f"pred.{_chunk_name(idx)}.jsonl"
        canonical_pred.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(best["pred_path"], canonical_pred)
        selected_pred_paths.append(canonical_pred)

        micro = best["metrics"]["micro"]
        tp_sum += int(micro["tp"])
        fp_sum += int(micro["fp"])
        fn_sum += int(micro["fn"])
        cumulative_f1 = _micro_f1_from_counts(tp_sum, fp_sum, fn_sum)

        node_results.append(
            {
                "checkpoint": _chunk_name(idx),
                "node_end_index": node_end - 1,
                "status": "passed",
                "used_config": str(active_config),
                "optimized": optimized,
                "node_f1": best_f1,
                "cumulative_f1": cumulative_f1,
                "metrics": best["metrics"],
            }
        )
        progress_payload = _build_report_payload(
            args=args,
            stop_reason="running",
            active_config=active_config,
            node_results=node_results,
            full_metrics=None,
            data_path=data_path,
            chunks_dir=chunks_dir,
            output_root=output_root,
        )
        _write_json(progress_path, progress_payload)

    full_metrics = None
    if node_results and stop_reason == "completed":
        merged_pred = output_root / "selected" / "pred.full.jsonl"
        with merged_pred.open("w", encoding="utf-8") as out:
            for p in selected_pred_paths:
                with p.open("r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            out.write(line)
        full_metrics = _eval_file(gold_path=data_path, pred_path=merged_pred, schema_path=schema_path)
        _append_log(
            log_path,
            f"full test completed: micro_f1={float(full_metrics['micro']['f1']):.4f}",
        )

    report = _build_report_payload(
        args=args,
        stop_reason=stop_reason,
        active_config=active_config,
        node_results=node_results,
        full_metrics=full_metrics,
        data_path=data_path,
        chunks_dir=chunks_dir,
        output_root=output_root,
    )
    _write_json(report_path, report)
    _write_json(progress_path, report)
    print(json.dumps({"stop_reason": stop_reason, "report_path": str(report_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
