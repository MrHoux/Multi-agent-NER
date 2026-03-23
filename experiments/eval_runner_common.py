from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from maner.core.schema import load_schema
from maner.eval.metrics import evaluate_from_files

from dataset_runtime import materialize_jsonl_window, split_jsonl_file


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(cwd))


def append_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"- {ts}: {message}\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def chunk_name(idx: int) -> str:
    return f"chunk{idx + 1:02d}"


def run_pipeline_once(
    *,
    repo_root: Path,
    config_path: Path,
    data_path: Path,
    schema_path: Path,
    pred_path: Path,
    reader_type: str = "generic_jsonl",
    prompt_overlay_path: Path | None = None,
    memory_db_path: Path | None = None,
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
        f"data.reader_type={reader_type}",
        "--set",
        f"output.predictions_path={pred_path}",
    ]
    if prompt_overlay_path is not None and prompt_overlay_path.exists():
        cmd.extend(["--set", f"prompts_path={prompt_overlay_path}"])
    if memory_db_path is not None:
        cmd.extend(["--set", f"memory.sqlite_path={memory_db_path}"])
    run_cmd(cmd, cwd=repo_root)


def eval_file(gold_path: Path, pred_path: Path, schema_path: Path) -> dict[str, Any]:
    schema = load_schema(schema_path)
    return evaluate_from_files(str(gold_path), str(pred_path), schema)


def merge_jsonl_files(paths: list[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for path in paths:
            with path.open("r", encoding="utf-8") as src:
                for line in src:
                    if line.strip():
                        out.write(line)


def micro_f1_from_counts(tp: int, fp: int, fn: int) -> float:
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    return (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0


def materialize_eval_window(
    *,
    dataset_name: str,
    split: str,
    source_data_path: Path,
    output_jsonl: Path,
    output_chunks_dir: Path,
    start_index: int,
    max_samples: int,
    chunk_size: int,
) -> tuple[Path, list[Path], int]:
    kept = materialize_jsonl_window(
        source_path=source_data_path,
        output_path=output_jsonl,
        start_index=start_index,
        max_samples=max_samples,
    )
    chunk_paths = split_jsonl_file(
        source_path=output_jsonl,
        chunks_dir=output_chunks_dir,
        dataset_name=dataset_name,
        split=split,
        chunk_size=chunk_size,
    )
    return output_jsonl, chunk_paths, kept
