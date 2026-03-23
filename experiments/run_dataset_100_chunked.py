from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval_runner_common import (
    append_log,
    chunk_name,
    eval_file,
    materialize_eval_window,
    merge_jsonl_files,
    run_pipeline_once,
)


def evaluate_config(
    *,
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
    reader_type: str,
    prompt_overlay_path: Path | None,
) -> tuple[bool, dict[str, Any]]:
    cfg_stem = config_path.stem
    cfg_out = output_root / cfg_stem
    cfg_out.mkdir(parents=True, exist_ok=True)
    memory_db_path = output_root / f"memory.{cfg_stem}.db"
    if memory_db_path.exists():
        memory_db_path.unlink()

    gate_scores: list[float] = []
    gate_metrics: dict[str, Any] = {}
    for idx in range(gate_chunk_count):
        chunk = chunk_paths[idx]
        pred_path = cfg_out / f"pred.{chunk_name(idx)}.jsonl"
        run_pipeline_once(
            repo_root=repo_root,
            config_path=config_path,
            data_path=chunk,
            schema_path=schema_path,
            pred_path=pred_path,
            reader_type=reader_type,
            prompt_overlay_path=prompt_overlay_path,
            memory_db_path=memory_db_path,
        )
        metrics = eval_file(chunk, pred_path, schema_path)
        f1 = float(metrics["micro"]["f1"])
        gate_scores.append(f1)
        gate_metrics[chunk_name(idx)] = metrics
        append_log(log_path, f"{cfg_stem} {chunk_name(idx)} micro_f1={f1:.4f}")

    gate_avg = sum(gate_scores) / max(1, len(gate_scores))
    append_log(log_path, f"{cfg_stem} first_two_avg_f1={gate_avg:.4f}")
    if gate_avg < threshold:
        return False, {
            "config": str(config_path),
            "status": "fail_threshold",
            "first_two_avg_f1": gate_avg,
            "first_two_metrics": gate_metrics,
        }

    chunk_metrics: dict[str, Any] = {}
    pred_paths: list[Path] = []
    for idx, chunk in enumerate(chunk_paths):
        pred_path = cfg_out / f"pred.{chunk_name(idx)}.jsonl"
        if idx >= gate_chunk_count:
            run_pipeline_once(
                repo_root=repo_root,
                config_path=config_path,
                data_path=chunk,
                schema_path=schema_path,
                pred_path=pred_path,
                reader_type=reader_type,
                prompt_overlay_path=prompt_overlay_path,
                memory_db_path=memory_db_path,
            )
        metrics = eval_file(chunk, pred_path, schema_path)
        f1 = float(metrics["micro"]["f1"])
        pred_paths.append(pred_path)
        chunk_metrics[chunk_name(idx)] = metrics
        append_log(log_path, f"{cfg_stem} {chunk_name(idx)} micro_f1={f1:.4f}")

    merged_pred = cfg_out / "pred.full100.jsonl"
    merge_jsonl_files(pred_paths, merged_pred)
    full_metrics = eval_file(merged_gold_path, merged_pred, schema_path)
    full_f1 = float(full_metrics["micro"]["f1"])
    append_log(log_path, f"{cfg_stem} full100 micro_f1={full_f1:.4f}")

    if full_f1 + drop_tolerance < gate_avg:
        append_log(
            log_path,
            (
                f"{cfg_stem} rejected for generalization_drop: "
                f"first_two_avg={gate_avg:.4f}, full100={full_f1:.4f}, "
                f"drop_tolerance={drop_tolerance:.4f}"
            ),
        )
        return False, {
            "config": str(config_path),
            "status": "fail_generalization_drop",
            "first_two_avg_f1": gate_avg,
            "full100_f1": full_f1,
            "chunk_metrics": chunk_metrics,
            "full_metrics": full_metrics,
        }

    return True, {
        "config": str(config_path),
        "status": "selected",
        "first_two_avg_f1": gate_avg,
        "full100_f1": full_f1,
        "chunk_metrics": chunk_metrics,
        "full_metrics": full_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run dataset-agnostic NER evaluation on a windowed sample set with chunk gates."
        )
    )
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--source_data_path", required=True)
    parser.add_argument("--schema_path", required=True)
    parser.add_argument("--reader_type", default="generic_jsonl")
    parser.add_argument("--prompt_overlay_path", default="")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--chunk_size", type=int, default=20)
    parser.add_argument("--gate_samples", type=int, default=40)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--drop_tolerance", type=float, default=0.08)
    parser.add_argument("--config_candidates", nargs="+", required=True)
    parser.add_argument("--log_path", required=True)
    parser.add_argument("--report_path", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    log_path = repo_root / args.log_path
    report_path = repo_root / args.report_path
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
    merged_gold_path = source_data_path.parent / f"{args.dataset_name}_{args.split}.{range_suffix}.jsonl"
    chunks_dir = source_data_path.parent / f"chunks_{range_suffix}"

    _, chunk_paths, kept = materialize_eval_window(
        dataset_name=args.dataset_name,
        split=args.split,
        source_data_path=source_data_path,
        output_jsonl=merged_gold_path,
        output_chunks_dir=chunks_dir,
        start_index=args.start_index,
        max_samples=args.max_samples,
        chunk_size=args.chunk_size,
    )
    if not chunk_paths:
        raise RuntimeError("No chunks generated for evaluation.")

    gate_chunk_count = max(1, min(len(chunk_paths), args.gate_samples // max(1, args.chunk_size)))

    append_log(
        log_path,
        (
            "start optimization cycle: "
            f"threshold={args.threshold:.2f}, drop_tolerance={args.drop_tolerance:.2f}, "
            f"start_index={args.start_index}, max_samples={args.max_samples}, "
            f"kept_samples={kept}, chunk_size={args.chunk_size}, "
            f"gate_samples={args.gate_samples}, gate_chunk_count={gate_chunk_count}"
        ),
    )

    all_results: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for cfg in args.config_candidates:
        cfg_path = repo_root / cfg
        if not cfg_path.exists():
            append_log(log_path, f"skip missing config: {cfg_path}")
            continue
        append_log(log_path, f"evaluate config: {cfg_path}")
        ok, result = evaluate_config(
            repo_root=repo_root,
            config_path=cfg_path,
            schema_path=schema_path,
            merged_gold_path=merged_gold_path,
            chunk_paths=chunk_paths,
            gate_chunk_count=gate_chunk_count,
            output_root=output_root,
            log_path=log_path,
            threshold=args.threshold,
            drop_tolerance=args.drop_tolerance,
            reader_type=args.reader_type,
            prompt_overlay_path=prompt_overlay_path,
        )
        all_results.append(result)
        if ok:
            selected = result
            append_log(log_path, f"selected config: {cfg_path}")
            break
        append_log(log_path, f"config rejected: {cfg_path} status={result.get('status')}")

    summary = {
        "threshold": args.threshold,
        "drop_tolerance": args.drop_tolerance,
        "start_index": args.start_index,
        "max_samples": args.max_samples,
        "kept_samples": kept,
        "chunk_size": args.chunk_size,
        "gate_samples": args.gate_samples,
        "gate_chunk_count": gate_chunk_count,
        "data_path": str(merged_gold_path),
        "results": all_results,
        "selected": selected,
    }
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if selected is None:
        append_log(log_path, "no config satisfied thresholds; optimization loop requires further changes")
        raise SystemExit("No configuration satisfied threshold/generalization checks.")

    print(json.dumps(selected, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
