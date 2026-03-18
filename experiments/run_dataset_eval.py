from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _to_repo_rel(repo_root: Path, target: Path) -> str:
    return str(target.resolve().relative_to(repo_root.resolve())).replace("\\", "/")


def _load_profile(profile_path: Path) -> Profile:
    repo_root = _repo_root()
    raw = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    paths = raw["paths"]
    return Profile(
        path=profile_path.resolve(),
        dataset_id=str(raw["dataset_id"]),
        runner_script=(repo_root / str(raw["runner_script"])).resolve(),
        configs_dir=(repo_root / str(paths["configs_dir"])).resolve(),
        results_dir=(repo_root / str(paths["results_dir"])).resolve(),
        logs_dir=(repo_root / str(paths["logs_dir"])).resolve(),
        runtime_dir=(repo_root / str(paths["runtime_dir"])).resolve(),
        execution=dict(raw["execution"]),
    )


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


def _build_start_command(profile: Profile, args: argparse.Namespace) -> tuple[list[str], dict[str, Path], dict[str, Any]]:
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
        "start_index": start_index,
        "max_samples": max_samples,
        "checkpoint_size": checkpoint_size,
        "threshold": threshold,
        "active_config": active_config,
        "optimization_configs": optimization_configs,
        "resume": resume,
    }
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


def _cmd_start(args: argparse.Namespace) -> int:
    profile = _load_profile(Path(args.profile))
    _ensure_dirs(profile)
    repo_root = _repo_root()

    command, paths, resolved_exec = _build_start_command(profile, args)
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
    start_index = int(exec_cfg["start_index"])
    max_samples = int(exec_cfg["max_samples"])
    checkpoint_size = int(exec_cfg["checkpoint_size"])
    range_suffix = f"{start_index}_{start_index + max_samples - 1}"

    chunks_dir = _repo_root() / "datasets" / profile.dataset_id / f"chunks_{range_suffix}_{checkpoint_size}"
    schema_path = _repo_root() / "datasets" / profile.dataset_id / f"schema.{profile.dataset_id}.json"
    selected_dir = profile.results_dir / "selected"
    pred_files = sorted(selected_dir.glob("pred.chunk*.jsonl"))

    summary: dict[str, Any] = {
        "completed_chunks": len(pred_files),
        "total_chunks": math.ceil(max_samples / checkpoint_size),
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
        gold_path = chunks_dir / f"conll2003_{exec_cfg['split']}.{_chunk_name(idx)}.jsonl"
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
    profile = _load_profile(Path(args.profile))
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
        completed = len([x for x in nodes if x.get("status") in {"passed", "resumed"}])
        total = math.ceil(int(exec_cfg["max_samples"]) / int(exec_cfg["checkpoint_size"]))
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
        completed = len([x for x in nodes if x.get("status") in {"passed", "resumed"}])
        total = math.ceil(int(exec_cfg["max_samples"]) / int(exec_cfg["checkpoint_size"]))
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
    profile = _load_profile(Path(args.profile))
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
    profile = _load_profile(Path(args.profile))
    log_path = profile.logs_dir / "console.log"
    return _tail_file(log_path, lines=int(args.lines), follow=bool(args.follow))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified dataset evaluation entrypoint")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_start = sub.add_parser("start", help="Start evaluation")
    p_start.add_argument("--profile", required=True)
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

    p_status = sub.add_parser("status", help="Show progress and accuracy")
    p_status.add_argument("--profile", required=True)

    p_pause = sub.add_parser("pause", help="Pause running process")
    p_pause.add_argument("--profile", required=True)

    p_resume = sub.add_parser("resume", help="Resume paused process")
    p_resume.add_argument("--profile", required=True)

    p_logs = sub.add_parser("logs", help="Show console log")
    p_logs.add_argument("--profile", required=True)
    p_logs.add_argument("--lines", type=int, default=50)
    p_logs.add_argument("--follow", action="store_true")

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
    raise SystemExit(2)


if __name__ == "__main__":
    main()
