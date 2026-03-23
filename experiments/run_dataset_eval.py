from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from dataset_runtime import (
    DatasetLayout,
    count_jsonl_records,
    ensure_dataset_assets,
    ensure_schema_stub,
    ensure_standard_profile,
    infer_dataset_layout,
    scaffold_source_manifest,
    validate_dataset_files,
)
from maner.core.schema import load_schema
from maner.eval.metrics import evaluate_from_files


@dataclass
class Profile:
    path: Path
    dataset_id: str
    runner_script: Path
    configs_dir: Path
    results_dir: Path
    logs_dir: Path
    runtime_dir: Path
    execution: dict[str, Any]
    prompt_generation: dict[str, Any]
    data: dict[str, Any]
    bootstrap: dict[str, Any]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _to_repo_rel(repo_root: Path, target: Path | str) -> str:
    path = Path(target)
    try:
        return str(path.resolve().relative_to(repo_root.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def _resolve_repo_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (repo_root / raw_path).resolve()


def _profile_from_raw(profile_path: Path, raw: dict[str, Any]) -> Profile:
    repo_root = _repo_root()
    paths = raw.get("paths", {})
    return Profile(
        path=profile_path.resolve(),
        dataset_id=str(raw["dataset_id"]),
        runner_script=_resolve_repo_path(repo_root, str(raw["runner_script"])),
        configs_dir=_resolve_repo_path(repo_root, str(paths.get("configs_dir", "configs"))),
        results_dir=_resolve_repo_path(repo_root, str(paths["results_dir"])),
        logs_dir=_resolve_repo_path(repo_root, str(paths["logs_dir"])),
        runtime_dir=_resolve_repo_path(repo_root, str(paths["runtime_dir"])),
        execution=dict(raw.get("execution", {})),
        prompt_generation=dict(raw.get("prompt_generation", {})),
        data=dict(raw.get("data", {})),
        bootstrap=dict(raw.get("bootstrap", {})),
    )


def _load_profile(profile_path: Path) -> Profile:
    raw = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    return _profile_from_raw(profile_path, raw)


def _resolve_profile(profile_arg: str | None, dataset_id_arg: str | None) -> Profile:
    if profile_arg:
        return _load_profile(Path(profile_arg))
    if not dataset_id_arg:
        raise ValueError("Provide either --profile or --dataset-id.")
    layout = infer_dataset_layout(_repo_root(), dataset_id_arg)
    ensure_standard_profile(layout)
    return _load_profile(layout.profile_path)


def _layout_for_profile(profile: Profile) -> DatasetLayout:
    return infer_dataset_layout(_repo_root(), profile.dataset_id)


def _ensure_dirs(profile: Profile) -> None:
    profile.results_dir.mkdir(parents=True, exist_ok=True)
    profile.logs_dir.mkdir(parents=True, exist_ok=True)
    profile.runtime_dir.mkdir(parents=True, exist_ok=True)


def _runtime_path(profile: Profile) -> Path:
    return profile.runtime_dir / "runtime.json"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _current_split_source(profile: Profile, exec_cfg: dict[str, Any]) -> Path:
    repo_root = _repo_root()
    split = str(exec_cfg.get("split", "test"))
    raw_path = str(exec_cfg.get("source_data_path", "")).strip()
    if raw_path:
        return _resolve_repo_path(repo_root, raw_path)
    split_paths = dict(profile.data.get("split_paths", {}))
    if split not in split_paths:
        raise FileNotFoundError(f"Missing split path for '{split}' in profile {profile.path}")
    return _resolve_repo_path(repo_root, str(split_paths[split]))


def _current_schema_path(profile: Profile, exec_cfg: dict[str, Any]) -> Path:
    repo_root = _repo_root()
    raw_path = str(exec_cfg.get("schema_path") or profile.data.get("schema_path", "")).strip()
    if not raw_path:
        raise FileNotFoundError(f"Missing schema_path in profile {profile.path}")
    return _resolve_repo_path(repo_root, raw_path)


def _current_reader_type(profile: Profile, exec_cfg: dict[str, Any]) -> str:
    return str(exec_cfg.get("reader_type") or profile.data.get("reader_type", "generic_jsonl"))


def _prompt_overlay_path(profile: Profile) -> Path | None:
    repo_root = _repo_root()
    raw_path = str(profile.prompt_generation.get("output_prompt_path", "")).strip()
    if not raw_path:
        return None
    return _resolve_repo_path(repo_root, raw_path)


def _ensure_profile_assets(profile: Profile, exec_cfg: dict[str, Any], *, require_train: bool = False) -> None:
    layout = _layout_for_profile(profile)
    required = [_current_schema_path(profile, exec_cfg), _current_split_source(profile, exec_cfg)]
    if require_train and profile.prompt_generation.get("enabled", False):
        train_path = str(profile.prompt_generation.get("train_data_path", "")).strip()
        if train_path:
            required.append(_resolve_repo_path(_repo_root(), train_path))
    ensure_dataset_assets(layout, required_paths=required)


def _build_start_command(
    profile: Profile,
    args: argparse.Namespace,
) -> tuple[list[str], dict[str, Path], dict[str, Any]]:
    repo_root = _repo_root()
    exec_cfg = dict(profile.execution)

    dataset_name = str(exec_cfg.get("dataset_name", profile.dataset_id))
    split = str(exec_cfg.get("split", "test"))
    start_index = int(args.start_index if args.start_index is not None else exec_cfg.get("start_index", 0))
    max_samples = int(args.max_samples if args.max_samples is not None else exec_cfg.get("max_samples", 0))
    checkpoint_size = int(
        args.checkpoint_size if args.checkpoint_size is not None else exec_cfg.get("checkpoint_size", 100)
    )
    threshold = float(args.threshold if args.threshold is not None else exec_cfg.get("threshold", 0.75))
    active_config = str(args.active_config if args.active_config else exec_cfg["active_config"])
    optimization_configs = list(args.optimization_configs or exec_cfg.get("optimization_configs", []))
    if not optimization_configs:
        optimization_configs = [active_config]
    resume = bool(exec_cfg.get("resume", False) if args.resume is None else args.resume)
    source_data_path = _current_split_source(profile, exec_cfg)
    schema_path = _current_schema_path(profile, exec_cfg)
    reader_type = _current_reader_type(profile, exec_cfg)
    prompt_overlay_path = _prompt_overlay_path(profile)

    report_path = profile.results_dir / "checkpoint_report.json"
    progress_path = profile.results_dir / "checkpoint_progress.json"
    protocol_log = profile.logs_dir / "EXECUTION_PROTOCOL_AND_LOG.md"
    console_log = profile.logs_dir / "console.log"

    command = [
        sys.executable,
        "-u",
        str(profile.runner_script),
        "--dataset_name",
        dataset_name,
        "--split",
        split,
        "--source_data_path",
        _to_repo_rel(repo_root, source_data_path),
        "--schema_path",
        _to_repo_rel(repo_root, schema_path),
        "--reader_type",
        reader_type,
        "--start_index",
        str(start_index),
        "--max_samples",
        str(max_samples),
        "--checkpoint_size",
        str(checkpoint_size),
        "--threshold",
        str(threshold),
        "--active_config",
        active_config,
        "--optimization_configs",
        *optimization_configs,
        "--report_path",
        _to_repo_rel(repo_root, report_path),
        "--progress_path",
        _to_repo_rel(repo_root, progress_path),
        "--log_path",
        _to_repo_rel(repo_root, protocol_log),
    ]
    if prompt_overlay_path is not None:
        command.extend(["--prompt_overlay_path", _to_repo_rel(repo_root, prompt_overlay_path)])
    if args.max_nodes and int(args.max_nodes) > 0:
        command.extend(["--max_nodes", str(int(args.max_nodes))])
    if resume:
        command.append("--resume")

    paths = {
        "report_path": report_path,
        "progress_path": progress_path,
        "protocol_log": protocol_log,
        "console_log": console_log,
    }
    resolved_exec = {
        "dataset_name": dataset_name,
        "split": split,
        "source_data_path": _to_repo_rel(repo_root, source_data_path),
        "schema_path": _to_repo_rel(repo_root, schema_path),
        "reader_type": reader_type,
        "start_index": start_index,
        "max_samples": max_samples,
        "checkpoint_size": checkpoint_size,
        "threshold": threshold,
        "active_config": active_config,
        "optimization_configs": optimization_configs,
        "resume": resume,
    }
    if prompt_overlay_path is not None:
        resolved_exec["prompt_overlay_path"] = _to_repo_rel(repo_root, prompt_overlay_path)
    return command, paths, resolved_exec


def _write_runtime(profile: Profile, payload: dict[str, Any]) -> None:
    _runtime_path(profile).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_runtime(profile: Profile) -> dict[str, Any] | None:
    p = _runtime_path(profile)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _stream_process(proc: subprocess.Popen[str], log_path: Path) -> int:
    with log_path.open("a", encoding="utf-8") as logf:
        if proc.stdout is None:
            return proc.wait()
        for line in proc.stdout:
            print(line, end="")
            logf.write(line)
            logf.flush()
    return proc.wait()


def _reset_results_dir(results_dir: Path) -> None:
    if not results_dir.exists():
        return
    for child in results_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _cmd_start(args: argparse.Namespace) -> int:
    profile = _resolve_profile(args.profile, args.dataset_id)
    _ensure_dirs(profile)
    _ensure_profile_assets(profile, dict(profile.execution), require_train=False)
    _maybe_generate_expert_prompt(profile, force=bool(args.refresh_expert_prompt))

    command, paths, resolved_exec = _build_start_command(profile, args)
    if not bool(resolved_exec.get("resume", False)):
        _reset_results_dir(profile.results_dir)
    repo_root = _repo_root()
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if args.background:
        with paths["console_log"].open("a", encoding="utf-8") as logf:
            logf.write(f"\n[{started_at}] START {' '.join(command)}\n")
            logf.flush()
            proc = subprocess.Popen(
                command,
                cwd=str(repo_root),
                stdout=logf,
                stderr=subprocess.STDOUT,
                text=True,
                preexec_fn=os.setsid,
            )
        runtime = {
            "dataset_id": profile.dataset_id,
            "pid": proc.pid,
            "pgid": os.getpgid(proc.pid),
            "started_at": started_at,
            "status": "running",
            "command": command,
            "profile": str(profile.path),
            "console_log": str(paths["console_log"]),
            "report_path": str(paths["report_path"]),
            "progress_path": str(paths["progress_path"]),
            "execution": resolved_exec,
        }
        _write_runtime(profile, runtime)
        print(json.dumps({"started": True, "pid": proc.pid, "pgid": runtime["pgid"], "mode": "background"}, ensure_ascii=False))
        return 0

    proc = subprocess.Popen(
        command,
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    runtime = {
        "dataset_id": profile.dataset_id,
        "pid": proc.pid,
        "pgid": None,
        "started_at": started_at,
        "status": "running_foreground",
        "command": command,
        "profile": str(profile.path),
        "console_log": str(paths["console_log"]),
        "report_path": str(paths["report_path"]),
        "progress_path": str(paths["progress_path"]),
        "execution": resolved_exec,
    }
    _write_runtime(profile, runtime)

    exit_code = _stream_process(proc, paths["console_log"])
    runtime["status"] = "completed" if exit_code == 0 else f"failed({exit_code})"
    runtime["ended_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _write_runtime(profile, runtime)
    return exit_code


def _chunk_name(idx: int) -> str:
    return f"chunk{idx + 1:02d}"


def _compute_cumulative_from_selected(profile: Profile, exec_cfg: dict[str, Any]) -> dict[str, Any]:
    start_index = int(exec_cfg.get("start_index", 0))
    max_samples = int(exec_cfg.get("max_samples", 0))
    checkpoint_size = int(exec_cfg.get("checkpoint_size", 100))
    total_samples = max_samples
    if total_samples <= 0:
        total_samples = count_jsonl_records(_current_split_source(profile, exec_cfg)) - start_index

    range_suffix = (
        f"{start_index}_{start_index + total_samples - 1}"
        if max_samples > 0
        else f"{start_index}_all"
    )
    dataset_name = str(exec_cfg.get("dataset_name", profile.dataset_id))
    split = str(exec_cfg.get("split", "test"))
    chunks_dir = _repo_root() / "datasets" / profile.dataset_id / f"chunks_{range_suffix}_{checkpoint_size}"
    schema_path = _current_schema_path(profile, exec_cfg)
    selected_dir = profile.results_dir / "selected"
    pred_files = sorted(selected_dir.glob("pred.chunk*.jsonl"))

    summary: dict[str, Any] = {
        "completed_chunks": len(pred_files),
        "total_chunks": math.ceil(total_samples / checkpoint_size) if checkpoint_size > 0 else 0,
        "covered_samples": len(pred_files) * checkpoint_size,
        "last_chunk_f1": None,
        "cumulative_f1": None,
    }
    if not pred_files or not schema_path.exists():
        return summary

    schema = load_schema(schema_path)
    tp_sum = fp_sum = fn_sum = 0
    last_f1 = None
    for idx, pred_path in enumerate(pred_files):
        gold_path = chunks_dir / f"{dataset_name}_{split}.{_chunk_name(idx)}.jsonl"
        if not gold_path.exists():
            break
        metrics = evaluate_from_files(str(gold_path), str(pred_path), schema)["micro"]
        tp_sum += int(metrics["tp"])
        fp_sum += int(metrics["fp"])
        fn_sum += int(metrics["fn"])
        last_f1 = float(metrics["f1"])

    precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) else 0.0
    recall = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) else 0.0
    cumulative_f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    summary["last_chunk_f1"] = last_f1
    summary["cumulative_f1"] = cumulative_f1
    return summary


def _cmd_status(args: argparse.Namespace) -> int:
    profile = _resolve_profile(args.profile, args.dataset_id)
    _ensure_dirs(profile)
    runtime = _read_runtime(profile)

    progress_path = profile.results_dir / "checkpoint_progress.json"
    report_path = profile.results_dir / "checkpoint_report.json"
    exec_cfg = dict(profile.execution)
    if runtime and "execution" in runtime:
        exec_cfg.update(runtime["execution"])

    running = False
    pid = None
    if runtime and "pid" in runtime:
        pid = int(runtime["pid"])
        running = _pid_alive(pid)

    print(f"dataset={profile.dataset_id}")
    print(f"running={running}")
    if pid is not None:
        print(f"pid={pid}")

    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        nodes = progress.get("node_results", [])
        stop_reason = progress.get("stop_reason")
        total_samples = int(exec_cfg.get("max_samples", 0))
        if total_samples <= 0:
            total_samples = count_jsonl_records(_current_split_source(profile, exec_cfg)) - int(exec_cfg.get("start_index", 0))
        completed = len([x for x in nodes if x.get("status") in {"passed", "resumed"}])
        total = math.ceil(total_samples / int(exec_cfg["checkpoint_size"]))
        print(f"completed_nodes={completed}/{total}")
        print(f"stop_reason={stop_reason}")
        if nodes:
            last = nodes[-1]
            print(f"last_checkpoint={last.get('checkpoint')}")
            if last.get("node_f1") is not None:
                print(f"last_node_f1={float(last['node_f1']):.4f}")
            if last.get("cumulative_f1") is not None:
                print(f"cumulative_f1={float(last['cumulative_f1']):.4f}")
        print(f"progress_path={progress_path}")
        return 0

    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        nodes = report.get("node_results", [])
        total_samples = int(exec_cfg.get("max_samples", 0))
        if total_samples <= 0:
            total_samples = count_jsonl_records(_current_split_source(profile, exec_cfg)) - int(exec_cfg.get("start_index", 0))
        completed = len([x for x in nodes if x.get("status") in {"passed", "resumed"}])
        total = math.ceil(total_samples / int(exec_cfg["checkpoint_size"]))
        print(f"completed_nodes={completed}/{total}")
        print(f"stop_reason={report.get('stop_reason')}")
        if nodes:
            last = nodes[-1]
            if last.get("node_f1") is not None:
                print(f"last_node_f1={float(last['node_f1']):.4f}")
            if last.get("cumulative_f1") is not None:
                print(f"cumulative_f1={float(last['cumulative_f1']):.4f}")
        print(f"report_path={report_path}")
        return 0

    summary = _compute_cumulative_from_selected(profile, exec_cfg)
    print(f"completed_nodes={summary['completed_chunks']}/{summary['total_chunks']}")
    print(f"covered_samples={summary['covered_samples']}")
    if summary["last_chunk_f1"] is not None:
        print(f"last_node_f1={summary['last_chunk_f1']:.4f}")
    if summary["cumulative_f1"] is not None:
        print(f"cumulative_f1={summary['cumulative_f1']:.4f}")
    return 0


def _signal_group_or_pid(runtime: dict[str, Any], sig: int) -> None:
    pgid = runtime.get("pgid")
    if pgid:
        os.killpg(int(pgid), sig)
        return
    os.kill(int(runtime["pid"]), sig)


def _cmd_pause_resume(args: argparse.Namespace, *, resume: bool) -> int:
    profile = _resolve_profile(args.profile, args.dataset_id)
    runtime = _read_runtime(profile)
    if not runtime or "pid" not in runtime:
        print("no runtime metadata found")
        return 1
    pid = int(runtime["pid"])
    if not _pid_alive(pid):
        print("target process is not running")
        return 1

    _signal_group_or_pid(runtime, signal.SIGCONT if resume else signal.SIGSTOP)
    runtime["status"] = "running" if resume else "paused"
    runtime["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _write_runtime(profile, runtime)
    print(json.dumps({"ok": True, "action": "resume" if resume else "pause", "pid": pid}, ensure_ascii=False))
    return 0


def _tail_file(path: Path, lines: int, follow: bool) -> int:
    if not path.exists():
        print(f"log file not found: {path}")
        return 1
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in content[-lines:]:
        print(line)
    if not follow:
        return 0

    with path.open("r", encoding="utf-8", errors="replace") as f:
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if line:
                print(line, end="")
            else:
                time.sleep(0.5)


def _cmd_logs(args: argparse.Namespace) -> int:
    profile = _resolve_profile(args.profile, args.dataset_id)
    log_path = profile.logs_dir / "console.log"
    return _tail_file(log_path, lines=int(args.lines), follow=bool(args.follow))


def _maybe_generate_expert_prompt(profile: Profile, force: bool = False) -> None:
    if not profile.prompt_generation or not bool(profile.prompt_generation.get("enabled", True)):
        return
    repo_root = _repo_root()
    schema_path_raw = str(profile.prompt_generation.get("schema_path", "")).strip()
    if not schema_path_raw:
        return
    schema_path = _resolve_repo_path(repo_root, schema_path_raw)
    if not schema_path.exists():
        return

    output_path = repo_root / str(profile.prompt_generation.get("output_prompt_path", ""))
    if output_path.exists() and not force:
        return

    command = [
        sys.executable,
        str(repo_root / "experiments" / "generate_dataset_expert_prompt.py"),
        "--profile",
        str(profile.path),
    ]
    if force:
        command.append("--force")
    subprocess.run(command, cwd=str(repo_root), check=True)


def _cmd_generate_prompt(args: argparse.Namespace) -> int:
    profile = _resolve_profile(args.profile, args.dataset_id)
    _ensure_dirs(profile)
    _ensure_profile_assets(profile, dict(profile.execution), require_train=False)
    _maybe_generate_expert_prompt(profile, force=bool(args.force))
    return 0


def _cmd_chunked_eval(args: argparse.Namespace) -> int:
    profile = _resolve_profile(args.profile, args.dataset_id)
    _ensure_dirs(profile)
    exec_cfg = dict(profile.execution)
    _ensure_profile_assets(profile, exec_cfg, require_train=False)
    _maybe_generate_expert_prompt(profile, force=bool(args.refresh_expert_prompt))

    repo_root = _repo_root()
    dataset_name = str(exec_cfg.get("dataset_name", profile.dataset_id))
    split = str(exec_cfg.get("split", "test"))
    start_index = int(args.start_index if args.start_index is not None else exec_cfg.get("start_index", 0))
    max_samples = int(args.max_samples if args.max_samples is not None else 100)
    chunk_size = int(args.chunk_size if args.chunk_size is not None else 20)
    gate_samples = int(args.gate_samples if args.gate_samples is not None else 40)
    threshold = float(args.threshold if args.threshold is not None else 0.75)
    drop_tolerance = float(args.drop_tolerance if args.drop_tolerance is not None else 0.08)
    config_candidates = list(args.config_candidates or exec_cfg.get("optimization_configs", []))
    if not config_candidates:
        config_candidates = [str(exec_cfg["active_config"])]

    source_data_path = _current_split_source(profile, exec_cfg)
    schema_path = _current_schema_path(profile, exec_cfg)
    reader_type = _current_reader_type(profile, exec_cfg)
    prompt_overlay_path = _prompt_overlay_path(profile)
    range_suffix = f"{start_index}_{start_index + max(1, max_samples) - 1}"
    report_path = profile.results_dir.parent / f"chunks_{range_suffix}" / "selection_report.json"
    log_path = profile.logs_dir / "EXECUTION_PROTOCOL_AND_LOG.md"

    command = [
        sys.executable,
        str(repo_root / "experiments" / "run_dataset_100_chunked.py"),
        "--dataset_name",
        dataset_name,
        "--split",
        split,
        "--source_data_path",
        _to_repo_rel(repo_root, source_data_path),
        "--schema_path",
        _to_repo_rel(repo_root, schema_path),
        "--reader_type",
        reader_type,
        "--start_index",
        str(start_index),
        "--max_samples",
        str(max_samples),
        "--chunk_size",
        str(chunk_size),
        "--gate_samples",
        str(gate_samples),
        "--threshold",
        str(threshold),
        "--drop_tolerance",
        str(drop_tolerance),
        "--config_candidates",
        *config_candidates,
        "--log_path",
        _to_repo_rel(repo_root, log_path),
        "--report_path",
        _to_repo_rel(repo_root, report_path),
    ]
    if prompt_overlay_path is not None:
        command.extend(["--prompt_overlay_path", _to_repo_rel(repo_root, prompt_overlay_path)])

    result = subprocess.run(command, cwd=str(repo_root))
    return int(result.returncode)


def _cmd_init_dataset(args: argparse.Namespace) -> int:
    layout = infer_dataset_layout(_repo_root(), args.dataset_id)
    layout.dataset_dir.mkdir(parents=True, exist_ok=True)
    layout.experiments_dir.mkdir(parents=True, exist_ok=True)
    ensure_schema_stub(layout, force=bool(args.force_schema))
    ensure_standard_profile(layout, force=bool(args.force))
    if args.with_source_template:
        scaffold_source_manifest(layout, force=bool(args.force))
    payload = {
        "dataset_id": layout.dataset_id,
        "schema_path": str(layout.schema_path),
        "profile_path": str(layout.profile_path),
    }
    if args.with_source_template:
        payload["source_manifest_path"] = str(layout.source_manifest_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _cmd_bootstrap_dataset(args: argparse.Namespace) -> int:
    layout = infer_dataset_layout(_repo_root(), args.dataset_id)
    profile = _resolve_profile(None, args.dataset_id)
    exec_cfg = dict(profile.execution)
    required = [_current_schema_path(profile, exec_cfg), _current_split_source(profile, exec_cfg)]
    if args.include_train and profile.prompt_generation.get("enabled", False):
        train_raw = str(profile.prompt_generation.get("train_data_path", "")).strip()
        if train_raw:
            required.append(_resolve_repo_path(_repo_root(), train_raw))
    ensure_dataset_assets(layout, required_paths=required)
    print(json.dumps({"bootstrapped": True, "dataset_id": args.dataset_id}, ensure_ascii=False))
    return 0


def _cmd_validate_dataset(args: argparse.Namespace) -> int:
    layout = infer_dataset_layout(_repo_root(), args.dataset_id)
    split_paths = {k: v for k, v in layout.split_paths.items() if v}
    result = validate_dataset_files(
        schema_path=layout.schema_path,
        split_paths=split_paths,
        require_train=bool(args.require_train),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok") else 1


def _add_target_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile")
    parser.add_argument("--dataset-id")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified dataset evaluation entrypoint")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_start = sub.add_parser("start", help="Start checkpoint evaluation")
    _add_target_args(p_start)
    p_start.add_argument("--background", action="store_true")
    p_start.add_argument("--resume", dest="resume", action="store_true", default=None)
    p_start.add_argument("--no-resume", dest="resume", action="store_false")
    p_start.add_argument("--max-nodes", type=int, default=0)
    p_start.add_argument("--start-index", type=int, default=None)
    p_start.add_argument("--max-samples", type=int, default=None)
    p_start.add_argument("--checkpoint-size", type=int, default=None)
    p_start.add_argument("--threshold", type=float, default=None)
    p_start.add_argument("--active-config", default="")
    p_start.add_argument("--optimization-configs", nargs="+", default=None)
    p_start.add_argument("--refresh-expert-prompt", action="store_true")

    p_status = sub.add_parser("status", help="Show progress and accuracy")
    _add_target_args(p_status)

    p_pause = sub.add_parser("pause", help="Pause running process")
    _add_target_args(p_pause)

    p_resume = sub.add_parser("resume", help="Resume paused process")
    _add_target_args(p_resume)

    p_logs = sub.add_parser("logs", help="Show console log")
    _add_target_args(p_logs)
    p_logs.add_argument("--lines", type=int, default=50)
    p_logs.add_argument("--follow", action="store_true")

    p_generate = sub.add_parser("generate-prompt", help="Generate dataset expert prompt overlay")
    _add_target_args(p_generate)
    p_generate.add_argument("--force", action="store_true")

    p_chunked = sub.add_parser("chunked-eval", help="Run chunked sample evaluation")
    _add_target_args(p_chunked)
    p_chunked.add_argument("--start-index", type=int, default=None)
    p_chunked.add_argument("--max-samples", type=int, default=100)
    p_chunked.add_argument("--chunk-size", type=int, default=20)
    p_chunked.add_argument("--gate-samples", type=int, default=40)
    p_chunked.add_argument("--threshold", type=float, default=None)
    p_chunked.add_argument("--drop-tolerance", type=float, default=0.08)
    p_chunked.add_argument("--config-candidates", nargs="+", default=None)
    p_chunked.add_argument("--refresh-expert-prompt", action="store_true")

    p_init = sub.add_parser("init-dataset", help="Scaffold standard dataset layout")
    p_init.add_argument("--dataset-id", required=True)
    p_init.add_argument("--force", action="store_true")
    p_init.add_argument("--force-schema", action="store_true")
    p_init.add_argument("--with-source-template", action="store_true")

    p_bootstrap = sub.add_parser("bootstrap-dataset", help="Download or prepare dataset assets")
    p_bootstrap.add_argument("--dataset-id", required=True)
    p_bootstrap.add_argument("--include-train", action="store_true")

    p_validate = sub.add_parser("validate-dataset", help="Validate dataset schema and JSONL files")
    p_validate.add_argument("--dataset-id", required=True)
    p_validate.add_argument("--require-train", action="store_true")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "start":
        raise SystemExit(_cmd_start(args))
    if args.cmd == "status":
        raise SystemExit(_cmd_status(args))
    if args.cmd == "pause":
        raise SystemExit(_cmd_pause_resume(args, resume=False))
    if args.cmd == "resume":
        raise SystemExit(_cmd_pause_resume(args, resume=True))
    if args.cmd == "logs":
        raise SystemExit(_cmd_logs(args))
    if args.cmd == "generate-prompt":
        raise SystemExit(_cmd_generate_prompt(args))
    if args.cmd == "chunked-eval":
        raise SystemExit(_cmd_chunked_eval(args))
    if args.cmd == "init-dataset":
        raise SystemExit(_cmd_init_dataset(args))
    if args.cmd == "bootstrap-dataset":
        raise SystemExit(_cmd_bootstrap_dataset(args))
    if args.cmd == "validate-dataset":
        raise SystemExit(_cmd_validate_dataset(args))
    raise SystemExit(2)


if __name__ == "__main__":
    main()
