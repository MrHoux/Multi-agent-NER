from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatasetLayout:
    repo_root: Path
    dataset_id: str
    dataset_dir: Path
    schema_path: Path
    split_paths: dict[str, Path]
    source_manifest_path: Path
    experiments_dir: Path
    prompts_dir: Path
    results_dir: Path
    logs_dir: Path
    runtime_dir: Path
    profile_path: Path


def infer_dataset_layout(repo_root: Path, dataset_id: str) -> DatasetLayout:
    dataset_dir = (repo_root / "datasets" / dataset_id).resolve()
    experiments_dir = (repo_root / "experiments" / "datasets" / dataset_id).resolve()
    split_paths = {
        split: discover_split_path(dataset_dir, dataset_id, split)
        for split in ("train", "dev", "test")
    }
    schema_path = discover_schema_path(dataset_dir, dataset_id)
    return DatasetLayout(
        repo_root=repo_root.resolve(),
        dataset_id=dataset_id,
        dataset_dir=dataset_dir,
        schema_path=schema_path,
        split_paths=split_paths,
        source_manifest_path=(dataset_dir / "dataset.source.yaml").resolve(),
        experiments_dir=experiments_dir,
        prompts_dir=(experiments_dir / "prompts").resolve(),
        results_dir=(experiments_dir / "results").resolve(),
        logs_dir=(experiments_dir / "logs").resolve(),
        runtime_dir=(experiments_dir / "runtime").resolve(),
        profile_path=(experiments_dir / "dataset.eval.yaml").resolve(),
    )


def discover_schema_path(dataset_dir: Path, dataset_id: str) -> Path:
    exact = dataset_dir / f"schema.{dataset_id}.json"
    if exact.exists():
        return exact.resolve()
    candidates = sorted(dataset_dir.glob("schema*.json"))
    if not candidates:
        return exact.resolve()
    return candidates[0].resolve()


def discover_split_path(dataset_dir: Path, dataset_id: str, split: str) -> Path:
    exact = dataset_dir / f"{dataset_id}_{split}.jsonl"
    if exact.exists():
        return exact.resolve()

    candidates = sorted(dataset_dir.glob(f"{dataset_id}_{split}*.jsonl"))
    if not candidates:
        return exact.resolve()
    return _pick_best_jsonl_candidate(candidates).resolve()


def discover_promptgen_train_path(layout: DatasetLayout) -> Path:
    dataset_id = layout.dataset_id
    dataset_dir = layout.dataset_dir
    preferred = [
        dataset_dir / f"{dataset_id}_train.promptgen.jsonl",
        dataset_dir / f"{dataset_id}_train.promptgen.40.jsonl",
        dataset_dir / f"{dataset_id}_train.jsonl",
    ]
    for path in preferred:
        if path.exists():
            return path.resolve()

    candidates = sorted(dataset_dir.glob(f"{dataset_id}_train*.jsonl"))
    if candidates:
        return _pick_best_jsonl_candidate(candidates).resolve()
    return preferred[-1].resolve()


def ensure_standard_profile(layout: DatasetLayout, *, force: bool = False) -> Path:
    raw = build_default_profile_payload(layout)
    layout.experiments_dir.mkdir(parents=True, exist_ok=True)
    if layout.profile_path.exists() and not force:
        return layout.profile_path
    layout.profile_path.write_text(
        yaml.safe_dump(raw, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    return layout.profile_path


def build_default_profile_payload(layout: DatasetLayout) -> dict[str, Any]:
    repo_root = layout.repo_root
    source_test_path = layout.split_paths["test"]
    return {
        "dataset_id": layout.dataset_id,
        "runner_script": "experiments/run_dataset_full_checkpoints.py",
        "paths": {
            "configs_dir": _repo_rel(repo_root, layout.experiments_dir / "configs"),
            "results_dir": _repo_rel(repo_root, layout.results_dir / "full_test"),
            "logs_dir": _repo_rel(repo_root, layout.logs_dir),
            "runtime_dir": _repo_rel(repo_root, layout.runtime_dir),
        },
        "data": {
            "dataset_dir": _repo_rel(repo_root, layout.dataset_dir),
            "schema_path": _repo_rel(repo_root, layout.schema_path),
            "reader_type": "generic_jsonl",
            "source_manifest_path": _repo_rel(repo_root, layout.source_manifest_path),
            "split_paths": {
                split: _repo_rel(repo_root, path)
                for split, path in layout.split_paths.items()
            },
        },
        "execution": {
            "dataset_name": layout.dataset_id,
            "split": "test",
            "source_data_path": _repo_rel(repo_root, source_test_path),
            "schema_path": _repo_rel(repo_root, layout.schema_path),
            "reader_type": "generic_jsonl",
            "start_index": 0,
            "max_samples": 0,
            "checkpoint_size": 100,
            "threshold": 0.75,
            "active_config": "configs/runtime.deepseek.all_agents.yaml",
            "optimization_configs": [
                "configs/runtime.deepseek.all_agents.yaml",
            ],
            "resume": True,
        },
        "prompt_generation": {
            "enabled": True,
            "llm_config_path": "configs/runtime.deepseek.all_agents.yaml",
            "schema_path": _repo_rel(repo_root, layout.schema_path),
            "output_prompt_path": _repo_rel(
                repo_root,
                layout.prompts_dir / "expert.generated.yaml",
            ),
            "metadata_path": _repo_rel(
                repo_root,
                layout.prompts_dir / "expert.generated.meta.json",
            ),
            "template_path": "configs/expert_prompt_generator.yaml",
        },
        "bootstrap": {
            "manifest_path": _repo_rel(repo_root, layout.source_manifest_path),
        },
    }


def ensure_dataset_assets(
    layout: DatasetLayout,
    *,
    required_paths: list[Path],
) -> None:
    missing = [path for path in required_paths if not path.exists()]
    if not missing:
        return
    if not layout.source_manifest_path.exists():
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Missing dataset assets: {missing_text}. "
            f"Provide the files or add bootstrap manifest: {layout.source_manifest_path}"
        )

    manifest = yaml.safe_load(layout.source_manifest_path.read_text(encoding="utf-8")) or {}
    _run_bootstrap_manifest(layout.repo_root, manifest)

    missing_after = [path for path in required_paths if not path.exists()]
    if missing_after:
        missing_text = ", ".join(str(path) for path in missing_after)
        raise FileNotFoundError(
            f"Dataset bootstrap finished but required assets are still missing: {missing_text}"
        )


def ensure_schema_stub(layout: DatasetLayout, *, force: bool = False) -> Path:
    if layout.schema_path.exists() and not force:
        return layout.schema_path
    layout.dataset_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_name": layout.dataset_id,
        "entity_types": [
            {
                "name": "ENTITY",
                "description": "Replace with an entity type name and its description.",
            }
        ],
        "relation_constraints": [],
    }
    layout.schema_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return layout.schema_path


def count_jsonl_records(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def validate_dataset_files(
    *,
    schema_path: Path,
    split_paths: dict[str, Path],
    require_train: bool = False,
) -> dict[str, Any]:
    issues: list[str] = []
    stats: dict[str, Any] = {"schema_path": str(schema_path), "splits": {}}

    if not schema_path.exists():
        issues.append(f"missing schema: {schema_path}")
        return {"ok": False, "issues": issues, "stats": stats}

    raw_schema = json.loads(schema_path.read_text(encoding="utf-8"))
    entity_types = raw_schema.get("entity_types", [])
    type_names = {str(item.get("name", "")).strip() for item in entity_types if isinstance(item, dict)}
    if not type_names:
        issues.append("schema has no entity type names")
    missing_descriptions = [
        str(item.get("name", "")).strip()
        for item in entity_types
        if isinstance(item, dict) and not str(item.get("description", "")).strip()
    ]
    if missing_descriptions:
        issues.append(
            "schema types missing descriptions: " + ", ".join(x for x in missing_descriptions if x)
        )

    required_splits = ["test"] + (["train"] if require_train else [])
    for split in required_splits:
        path = split_paths.get(split)
        if path is None or not path.exists():
            issues.append(f"missing split: {split}")
            continue
        split_stats = _validate_jsonl_file(path, type_names)
        stats["splits"][split] = split_stats
        for item in split_stats.get("issues", []):
            issues.append(f"{split}: {item}")

    return {"ok": not issues, "issues": issues, "stats": stats}


def materialize_jsonl_window(
    *,
    source_path: Path,
    output_path: Path,
    start_index: int,
    max_samples: int,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    end_index = None if max_samples <= 0 else start_index + max_samples
    with source_path.open("r", encoding="utf-8") as src, output_path.open(
        "w",
        encoding="utf-8",
    ) as out:
        for idx, line in enumerate(src):
            if not line.strip():
                continue
            if idx < start_index:
                continue
            if end_index is not None and idx >= end_index:
                break
            out.write(line)
            kept += 1
    return kept


def split_jsonl_file(
    *,
    source_path: Path,
    chunks_dir: Path,
    dataset_name: str,
    split: str,
    chunk_size: int,
) -> list[Path]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths: list[Path] = []
    current_chunk_index = 0
    current_chunk_count = 0
    current_file = None

    def open_chunk(idx: int) -> tuple[Path, Any]:
        path = chunks_dir / f"{dataset_name}_{split}.chunk{idx + 1:02d}.jsonl"
        return path, path.open("w", encoding="utf-8")

    try:
        with source_path.open("r", encoding="utf-8") as src:
            for line in src:
                if not line.strip():
                    continue
                if current_file is None or current_chunk_count >= chunk_size:
                    if current_file is not None:
                        current_file.close()
                    chunk_path, current_file = open_chunk(current_chunk_index)
                    chunk_paths.append(chunk_path)
                    current_chunk_index += 1
                    current_chunk_count = 0
                current_file.write(line)
                current_chunk_count += 1
    finally:
        if current_file is not None:
            current_file.close()
    return chunk_paths


def load_schema_dataset_name(schema_path: Path) -> str:
    if not schema_path.exists():
        return ""
    raw = json.loads(schema_path.read_text(encoding="utf-8"))
    return str(raw.get("dataset_name") or raw.get("dataset_id") or "").strip()


def _pick_best_jsonl_candidate(candidates: list[Path]) -> Path:
    scored = [(count_jsonl_records(path), -len(path.name), path) for path in candidates]
    scored.sort(reverse=True)
    return scored[0][2]


def _repo_rel(repo_root: Path, target: Path) -> str:
    try:
        return str(target.resolve().relative_to(repo_root.resolve())).replace("\\", "/")
    except Exception:
        return str(target).replace("\\", "/")


def _run_bootstrap_manifest(repo_root: Path, manifest: dict[str, Any]) -> None:
    commands = manifest.get("commands")
    if commands is None and manifest.get("command"):
        commands = [manifest.get("command")]
    if commands:
        cwd = repo_root
        raw_cwd = str(manifest.get("cwd", "")).strip()
        if raw_cwd:
            cwd = (repo_root / raw_cwd).resolve()
        for item in commands:
            if isinstance(item, list):
                cmd = [str(x) for x in item]
                subprocess.run(cmd, cwd=str(cwd), check=True)
            else:
                subprocess.run(str(item), cwd=str(cwd), shell=True, check=True)

    downloads = manifest.get("downloads", [])
    if isinstance(downloads, dict):
        downloads = [downloads]
    for item in downloads:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url", "")).strip()
        output_path = str(item.get("output_path", "")).strip()
        if not url or not output_path:
            continue
        target = (repo_root / output_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as resp, target.open("wb") as out:
            out.write(resp.read())
        sha256 = str(item.get("sha256", "")).strip()
        if sha256:
            actual = hashlib.sha256(target.read_bytes()).hexdigest()
            if actual.lower() != sha256.lower():
                raise RuntimeError(
                    f"Downloaded file hash mismatch for {target}: {actual} != {sha256}"
                )

    huggingface = manifest.get("huggingface")
    if isinstance(huggingface, dict):
        _bootstrap_from_huggingface(repo_root, huggingface)


def scaffold_source_manifest(layout: DatasetLayout, *, force: bool = False) -> Path:
    if layout.source_manifest_path.exists() and not force:
        return layout.source_manifest_path
    layout.dataset_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "commands": [],
        "downloads": [],
        "huggingface": {},
        "examples": {
            "downloads": [
                {
                    "url": "https://example.com/dataset.jsonl",
                    "output_path": f"datasets/{layout.dataset_id}/{layout.dataset_id}_test.jsonl",
                }
            ],
            "huggingface": {
                "dataset_name": "conll2003",
                "format": "token_classification",
                "split_map": {
                    "train": "train",
                    "dev": "validation",
                    "test": "test",
                },
                "output_files": {
                    "train": f"datasets/{layout.dataset_id}/{layout.dataset_id}_train.jsonl",
                    "dev": f"datasets/{layout.dataset_id}/{layout.dataset_id}_dev.jsonl",
                    "test": f"datasets/{layout.dataset_id}/{layout.dataset_id}_test.jsonl",
                },
                "tokens_field": "tokens",
                "tags_field": "ner_tags",
                "sample_id_field": "id",
                "label_list": ["O", "B-ENTITY", "I-ENTITY"],
            },
        },
    }
    layout.source_manifest_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    return layout.source_manifest_path


def _validate_jsonl_file(path: Path, valid_types: set[str]) -> dict[str, Any]:
    issues: list[str] = []
    sample_count = 0
    mention_count = 0
    bad_type_count = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            sample_count += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                issues.append(f"invalid json at line {line_no}: {exc}")
                continue
            if not str(obj.get("id", "")).strip():
                issues.append(f"missing id at line {line_no}")
            text = obj.get("text")
            if not isinstance(text, str):
                issues.append(f"missing text at line {line_no}")
                continue
            mentions = obj.get("gold_mentions", [])
            if mentions is None:
                mentions = []
            if not isinstance(mentions, list):
                issues.append(f"gold_mentions is not list at line {line_no}")
                continue
            mention_count += len(mentions)
            for mention in mentions:
                if not isinstance(mention, dict):
                    issues.append(f"non-dict mention at line {line_no}")
                    continue
                start = int(mention.get("start", -1))
                end = int(mention.get("end", -1))
                ent_type = str(mention.get("ent_type", "")).strip()
                if not (0 <= start <= end <= len(text)):
                    issues.append(f"invalid offsets at line {line_no}")
                    continue
                if not ent_type:
                    issues.append(f"missing ent_type at line {line_no}")
                    continue
                if valid_types and ent_type not in valid_types:
                    bad_type_count += 1
    if bad_type_count:
        issues.append(f"{bad_type_count} mentions use types outside schema")
    return {
        "path": str(path),
        "sample_count": sample_count,
        "mention_count": mention_count,
        "issues": issues,
    }


def _bootstrap_from_huggingface(repo_root: Path, cfg: dict[str, Any]) -> None:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            "HuggingFace bootstrap requested but `datasets` is not installed."
        ) from exc

    dataset_name = str(cfg.get("dataset_name", "")).strip()
    if not dataset_name:
        raise ValueError("huggingface.dataset_name is required")
    subset = cfg.get("subset")
    tokens_field = str(cfg.get("tokens_field", "tokens"))
    tags_field = str(cfg.get("tags_field", "ner_tags"))
    sample_id_field = str(cfg.get("sample_id_field", "id"))
    label_list = cfg.get("label_list")
    split_map = dict(cfg.get("split_map", {}))
    output_files = dict(cfg.get("output_files", {}))
    fmt = str(cfg.get("format", "token_classification")).strip()

    if fmt != "token_classification":
        raise ValueError(f"Unsupported huggingface format: {fmt}")
    if not split_map or not output_files:
        raise ValueError("huggingface.split_map and huggingface.output_files are required")

    ds = load_dataset(dataset_name, subset) if subset not in {"", None} else load_dataset(dataset_name)
    for local_split, remote_split in split_map.items():
        if local_split not in output_files:
            continue
        if remote_split not in ds:
            raise KeyError(f"Remote split '{remote_split}' not found in dataset '{dataset_name}'")
        target = (repo_root / str(output_files[local_split])).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        records = ds[remote_split]
        inferred_labels = label_list
        if not inferred_labels:
            feature = records.features[tags_field]
            names = getattr(getattr(feature, "feature", None), "names", None)
            if names:
                inferred_labels = list(names)
        if not inferred_labels:
            raise ValueError("Unable to infer label_list; set huggingface.label_list explicitly")
        with target.open("w", encoding="utf-8") as out:
            for idx, item in enumerate(records):
                tokens = [str(x) for x in item[tokens_field]]
                tag_ids = item[tags_field]
                tag_names = [str(inferred_labels[int(tag)]) for tag in tag_ids]
                text, mentions = _bio_tokens_to_jsonl(tokens, tag_names)
                sample_id = str(item.get(sample_id_field, f"{local_split}-{idx:05d}"))
                payload = {
                    "id": sample_id,
                    "text": text,
                    "gold_mentions": mentions,
                }
                out.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _bio_tokens_to_jsonl(tokens: list[str], tag_names: list[str]) -> tuple[str, list[dict[str, Any]]]:
    positions: list[tuple[int, int]] = []
    cursor = 0
    parts: list[str] = []
    for token in tokens:
        if parts:
            parts.append(" ")
            cursor += 1
        start = cursor
        parts.append(token)
        cursor += len(token)
        positions.append((start, cursor))
    text = "".join(parts)

    mentions: list[dict[str, Any]] = []
    current_type = ""
    current_start = -1
    current_end = -1

    def flush() -> None:
        nonlocal current_type, current_start, current_end
        if current_type and current_start >= 0 and current_end >= current_start:
            mentions.append(
                {
                    "start": current_start,
                    "end": current_end,
                    "ent_type": current_type,
                    "text": text[current_start:current_end],
                }
            )
        current_type = ""
        current_start = -1
        current_end = -1

    for idx, tag in enumerate(tag_names):
        start, end = positions[idx]
        if tag == "O":
            flush()
            continue
        prefix, _, ent_type = tag.partition("-")
        prefix = prefix.upper()
        ent_type = ent_type.strip()
        if prefix == "B" or not current_type or ent_type != current_type:
            flush()
            current_type = ent_type
            current_start = start
            current_end = end
            continue
        if prefix == "I":
            current_end = end
            continue
        flush()
    flush()
    return text, mentions
