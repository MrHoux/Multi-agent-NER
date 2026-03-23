from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from eval_runner_common import (
    append_log,
    chunk_name,
    eval_file,
    materialize_eval_window,
    merge_jsonl_files,
    micro_f1_from_counts,
    run_pipeline_once,
    write_json,
)


def build_report_payload(
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


def choose_best_config_for_chunk(
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
    reader_type: str,
    prompt_overlay_path: Path | None,
) -> dict[str, Any]:
    candidates = [active_config] + [cfg for cfg in candidate_configs if cfg != active_config]
    best: dict[str, Any] | None = None

    for cfg in candidates:
        cfg_stem = cfg.stem
        cfg_out = output_root / cfg_stem
        pred_path = cfg_out / f"pred.{chunk_name(chunk_idx)}.jsonl"
        memory_db_path = output_root / f"memory.{cfg_stem}.db"
        run_pipeline_once(
            repo_root=repo_root,
            config_path=cfg,
            data_path=chunk_path,
            schema_path=schema_path,
            pred_path=pred_path,
            reader_type=reader_type,
            prompt_overlay_path=prompt_overlay_path,
            memory_db_path=memory_db_path,
        )
        metrics = eval_file(chunk_path, pred_path, schema_path)
        f1 = float(metrics["micro"]["f1"])
        append_log(log_path, f"{cfg_stem} {chunk_name(chunk_idx)} checkpoint_f1={f1:.4f}")
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


def _clear_memory_dbs(output_root: Path, config_paths: list[Path]) -> None:
    for cfg in config_paths:
        memory_db_path = output_root / f"memory.{cfg.stem}.db"
        if memory_db_path.exists():
            memory_db_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full-dataset checkpoint evaluation with dataset-agnostic JSONL input."
    )
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--source_data_path", required=True)
    parser.add_argument("--schema_path", required=True)
    parser.add_argument("--reader_type", default="generic_jsonl")
    parser.add_argument("--prompt_overlay_path", default="")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--checkpoint_size", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--active_config", required=True)
    parser.add_argument("--optimization_configs", nargs="+", required=True)
    parser.add_argument("--log_path", required=True)
    parser.add_argument("--report_path", required=True)
    parser.add_argument("--progress_path", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_nodes", type=int, default=0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    log_path = repo_root / args.log_path
    report_path = repo_root / args.report_path
    progress_path = repo_root / args.progress_path
    output_root = report_path.parent
    output_root.mkdir(parents=True, exist_ok=True)

    source_data_path = repo_root / args.source_data_path
    schema_path = repo_root / args.schema_path
    prompt_overlay_path = (repo_root / args.prompt_overlay_path) if args.prompt_overlay_path else None

    range_suffix = (
        f"{args.start_index}_{args.start_index + args.max_samples - 1}"
        if args.max_samples > 0
        else f"{args.start_index}_all"
    )
    data_path = source_data_path.parent / f"{args.dataset_name}_{args.split}.{range_suffix}.jsonl"
    chunks_dir = source_data_path.parent / f"chunks_{range_suffix}_{args.checkpoint_size}"

    _, chunk_paths, kept = materialize_eval_window(
        dataset_name=args.dataset_name,
        split=args.split,
        source_data_path=source_data_path,
        output_jsonl=data_path,
        output_chunks_dir=chunks_dir,
        start_index=args.start_index,
        max_samples=args.max_samples,
        chunk_size=args.checkpoint_size,
    )
    if not chunk_paths:
        raise RuntimeError("No chunks generated for checkpoint evaluation.")
    if args.max_nodes > 0:
        chunk_paths = chunk_paths[: args.max_nodes]

    append_log(
        log_path,
        (
            "start full-test checkpoint run: "
            f"start_index={args.start_index}, max_samples={args.max_samples}, "
            f"kept_samples={kept}, checkpoint_size={args.checkpoint_size}, "
            f"threshold={args.threshold:.2f}"
        ),
    )

    active_config = (repo_root / args.active_config).resolve()
    if not active_config.exists():
        raise FileNotFoundError(f"Active config not found: {active_config}")
    optimization_configs = [(repo_root / item).resolve() for item in args.optimization_configs]
    optimization_configs = [item for item in optimization_configs if item.exists()]
    if not optimization_configs:
        raise RuntimeError("No valid optimization configs found.")
    if not args.resume:
        _clear_memory_dbs(output_root, [active_config, *optimization_configs])

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
            existing_pred = selected_dir / f"pred.{chunk_name(idx)}.jsonl"
            if not existing_pred.exists():
                break
            metrics = eval_file(chunk_path, existing_pred, schema_path)
            micro = metrics["micro"]
            tp_sum += int(micro["tp"])
            fp_sum += int(micro["fp"])
            fn_sum += int(micro["fn"])
            cumulative_f1 = micro_f1_from_counts(tp_sum, fp_sum, fn_sum)
            node_results.append(
                {
                    "checkpoint": chunk_name(idx),
                    "node_end_index": args.start_index + (idx + 1) * args.checkpoint_size - 1,
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
            append_log(log_path, f"resume enabled: skipping {resume_start_idx} completed checkpoints")

    progress_payload = build_report_payload(
        args=args,
        stop_reason="running",
        active_config=active_config,
        node_results=node_results,
        full_metrics=None,
        data_path=data_path,
        chunks_dir=chunks_dir,
        output_root=output_root,
    )
    write_json(progress_path, progress_payload)

    for idx in range(resume_start_idx, len(chunk_paths)):
        chunk_path = chunk_paths[idx]
        node_end = args.start_index + (idx + 1) * args.checkpoint_size
        append_log(
            log_path,
            f"checkpoint {chunk_name(idx)} (up to sample_index {node_end - 1}) active_config={active_config}",
        )
        best = choose_best_config_for_chunk(
            repo_root=repo_root,
            output_root=output_root,
            schema_path=schema_path,
            chunk_path=chunk_path,
            chunk_idx=idx,
            active_config=active_config,
            candidate_configs=optimization_configs,
            threshold=args.threshold,
            log_path=log_path,
            reader_type=args.reader_type,
            prompt_overlay_path=prompt_overlay_path,
        )
        best_f1 = float(best["f1"])
        chosen_config = Path(str(best["config"]))
        optimized = chosen_config != active_config

        if best_f1 < args.threshold:
            stop_reason = f"checkpoint_below_threshold_unrecoverable:{chunk_name(idx)}:{best_f1:.4f}"
            append_log(log_path, f"stop at {chunk_name(idx)}: best_f1={best_f1:.4f} < threshold={args.threshold:.2f}")
            node_results.append(
                {
                    "checkpoint": chunk_name(idx),
                    "node_end_index": node_end - 1,
                    "status": "failed",
                    "active_config": str(active_config),
                    "best_config": str(chosen_config),
                    "node_f1": best_f1,
                    "metrics": best["metrics"],
                }
            )
            progress_payload = build_report_payload(
                args=args,
                stop_reason=stop_reason,
                active_config=active_config,
                node_results=node_results,
                full_metrics=None,
                data_path=data_path,
                chunks_dir=chunks_dir,
                output_root=output_root,
            )
            write_json(progress_path, progress_payload)
            break

        active_config = chosen_config
        if optimized:
            append_log(log_path, f"optimized at {chunk_name(idx)}: switch active_config -> {active_config}")

        canonical_pred = output_root / "selected" / f"pred.{chunk_name(idx)}.jsonl"
        canonical_pred.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(best["pred_path"], canonical_pred)
        selected_pred_paths.append(canonical_pred)

        micro = best["metrics"]["micro"]
        tp_sum += int(micro["tp"])
        fp_sum += int(micro["fp"])
        fn_sum += int(micro["fn"])
        cumulative_f1 = micro_f1_from_counts(tp_sum, fp_sum, fn_sum)
        node_results.append(
            {
                "checkpoint": chunk_name(idx),
                "node_end_index": node_end - 1,
                "status": "passed",
                "used_config": str(active_config),
                "optimized": optimized,
                "node_f1": best_f1,
                "cumulative_f1": cumulative_f1,
                "metrics": best["metrics"],
            }
        )
        progress_payload = build_report_payload(
            args=args,
            stop_reason="running",
            active_config=active_config,
            node_results=node_results,
            full_metrics=None,
            data_path=data_path,
            chunks_dir=chunks_dir,
            output_root=output_root,
        )
        write_json(progress_path, progress_payload)

    full_metrics = None
    if node_results and stop_reason == "completed":
        merged_pred = output_root / "selected" / "pred.full.jsonl"
        merge_jsonl_files(selected_pred_paths, merged_pred)
        full_metrics = eval_file(data_path, merged_pred, schema_path)
        append_log(log_path, f"full test completed: micro_f1={float(full_metrics['micro']['f1']):.4f}")

    report = build_report_payload(
        args=args,
        stop_reason=stop_reason,
        active_config=active_config,
        node_results=node_results,
        full_metrics=full_metrics,
        data_path=data_path,
        chunks_dir=chunks_dir,
        output_root=output_root,
    )
    write_json(report_path, report)
    write_json(progress_path, report)
    print(json.dumps({"stop_reason": stop_reason, "report_path": str(report_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
