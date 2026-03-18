from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _import_hf_load_dataset():
    """Import HuggingFace datasets.load_dataset, avoiding local ./datasets shadowing."""
    repo_root = Path(__file__).resolve().parents[2]
    original_sys_path = list(sys.path)

    try:
        for key in ["datasets"]:
            module = sys.modules.get(key)
            if module is None:
                continue
            module_file = getattr(module, "__file__", "") or ""
            if module_file and Path(module_file).resolve().is_relative_to(repo_root):
                del sys.modules[key]

        filtered: list[str] = []
        for entry in original_sys_path:
            if not entry:
                continue
            try:
                if Path(entry).resolve() == repo_root:
                    continue
            except OSError:
                pass
            filtered.append(entry)
        sys.path = filtered

        from datasets import load_dataset  # type: ignore

        return load_dataset
    finally:
        sys.path = original_sys_path


def _normalize_token(token: str) -> str:
    token_map = {
        "``": '"',
        "''": '"',
        "-LRB-": "(",
        "-RRB-": ")",
        "-LSB-": "[",
        "-RSB-": "]",
        "-LCB-": "{",
        "-RCB-": "}",
    }
    return token_map.get(token, token)


def _needs_space(prev_token: str | None, cur_token: str) -> bool:
    if prev_token is None:
        return False

    no_space_before = {
        ".",
        ",",
        ":",
        ";",
        "!",
        "?",
        "%",
        ")",
        "]",
        "}",
        "'s",
        "'re",
        "'ve",
        "'ll",
        "'d",
        "'m",
        "n't",
    }
    no_space_after = {"(", "[", "{", "$", "#", '"', "/"}

    cur_lower = cur_token.lower()
    if cur_token in no_space_before or cur_lower in no_space_before:
        return False
    if prev_token in no_space_after:
        return False
    if cur_token.startswith("'"):
        return False
    if prev_token.endswith("/"):
        return False
    return True


def _reconstruct_text_and_offsets(tokens: list[str]) -> tuple[str, list[int], list[int]]:
    starts: list[int] = []
    ends: list[int] = []
    chunks: list[str] = []
    cursor = 0

    prev: str | None = None
    for raw_token in tokens:
        token = _normalize_token(raw_token)
        if _needs_space(prev, token):
            chunks.append(" ")
            cursor += 1

        starts.append(cursor)
        chunks.append(token)
        cursor += len(token)
        ends.append(cursor)
        prev = token

    return "".join(chunks), starts, ends


def _decode_label(tag: int | str, label_names: list[str]) -> str:
    if isinstance(tag, int):
        return label_names[tag]
    return str(tag)


def _bio_to_mentions(
    tags: list[int | str],
    starts: list[int],
    ends: list[int],
    text: str,
    label_names: list[str],
) -> list[dict[str, Any]]:
    mentions: list[dict[str, Any]] = []
    current_type: str | None = None
    current_start: int | None = None
    current_end: int | None = None

    def flush_current() -> None:
        nonlocal current_type, current_start, current_end
        if current_type is None or current_start is None or current_end is None:
            current_type = None
            current_start = None
            current_end = None
            return
        mentions.append(
            {
                "start": current_start,
                "end": current_end,
                "ent_type": current_type,
                "text": text[current_start:current_end],
            }
        )
        current_type = None
        current_start = None
        current_end = None

    for i, raw_tag in enumerate(tags):
        label = _decode_label(raw_tag, label_names)
        if label == "O":
            flush_current()
            continue

        if "-" in label:
            prefix, ent_type = label.split("-", 1)
            prefix = prefix.upper()
        else:
            prefix = "B"
            ent_type = label

        if prefix == "B":
            flush_current()
            current_type = ent_type
            current_start = starts[i]
            current_end = ends[i]
            continue

        if prefix == "I":
            if current_type == ent_type and current_start is not None:
                current_end = ends[i]
            else:
                flush_current()
                current_type = ent_type
                current_start = starts[i]
                current_end = ends[i]
            continue

        flush_current()

    flush_current()
    return mentions


def _conll2003_schema() -> dict[str, Any]:
    return {
        "entity_types": [
            {
                "name": "PER",
                "description": "Person names and named individuals, including full names and surnames.",
            },
            {
                "name": "ORG",
                "description": "Organizations such as companies, agencies, institutions, sports teams, and political groups.",
            },
            {
                "name": "LOC",
                "description": "Named locations including countries, cities, regions, and other geographic places.",
            },
            {
                "name": "MISC",
                "description": "Named entities not covered by PER/ORG/LOC, such as nationalities, events, works, and products.",
            },
        ],
        "relation_constraints": [],
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_conll2003(
    dataset_name: str,
    split: str,
    start_index: int,
    max_samples: int,
    chunk_size: int,
    output_jsonl: Path,
    output_schema: Path,
    output_chunks_dir: Path,
) -> tuple[Path, Path, list[Path]]:
    if max_samples <= 0:
        raise ValueError("max_samples must be > 0")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if start_index < 0:
        raise ValueError("start_index must be >= 0")

    load_dataset = _import_hf_load_dataset()
    ds = load_dataset(dataset_name, trust_remote_code=True)
    if split not in ds:
        raise ValueError(f"Split '{split}' not in dataset. Available: {list(ds.keys())}")

    split_ds = ds[split]
    source_start = min(start_index, len(split_ds))
    source_end = min(source_start + max_samples, len(split_ds))
    split_ds = split_ds.select(range(source_start, source_end))

    feature = split_ds.features["ner_tags"].feature
    label_names = list(getattr(feature, "names", []))
    if not label_names:
        raise ValueError("Failed to read label names from ner_tags feature.")

    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(split_ds):
        source_idx = source_start + idx
        tokens = [str(t) for t in row["tokens"]]
        tags = list(row["ner_tags"])
        text, starts, ends = _reconstruct_text_and_offsets(tokens)
        gold_mentions = _bio_to_mentions(tags, starts, ends, text, label_names)
        rows.append(
            {
                "id": f"{split}-{source_idx:05d}",
                "text": text,
                "gold_mentions": gold_mentions,
            }
        )

    _write_jsonl(output_jsonl, rows)

    output_schema.parent.mkdir(parents=True, exist_ok=True)
    with output_schema.open("w", encoding="utf-8") as f:
        json.dump(_conll2003_schema(), f, ensure_ascii=False, indent=2)

    output_chunks_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths: list[Path] = []
    chunk_count = (len(rows) + chunk_size - 1) // chunk_size
    for chunk_idx in range(chunk_count):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min((chunk_idx + 1) * chunk_size, len(rows))
        chunk_rows = rows[chunk_start:chunk_end]
        chunk_path = output_chunks_dir / f"conll2003_{split}.chunk{chunk_idx + 1:02d}.jsonl"
        _write_jsonl(chunk_path, chunk_rows)
        chunk_paths.append(chunk_path)

    print(f"Wrote data: {output_jsonl}")
    print(f"Wrote schema: {output_schema}")
    print(f"Wrote {len(chunk_paths)} chunks to: {output_chunks_dir}")
    print(
        f"Samples: {len(rows)} | Range: [{source_start}, {source_end}) | Chunk size: {chunk_size}"
    )
    return output_jsonl, output_schema, chunk_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CoNLL2003 jsonl + schema for MANER pipeline")
    parser.add_argument("--dataset_name", default="conll2003")
    parser.add_argument("--split", default="test")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--chunk_size", type=int, default=20)
    parser.add_argument("--output_jsonl", default="datasets/conll2003/conll2003_test.100.jsonl")
    parser.add_argument("--output_schema", default="datasets/conll2003/schema.conll2003.json")
    parser.add_argument(
        "--output_chunks_dir",
        default="datasets/conll2003/chunks100",
    )
    args = parser.parse_args()

    prepare_conll2003(
        dataset_name=args.dataset_name,
        split=args.split,
        start_index=args.start_index,
        max_samples=args.max_samples,
        chunk_size=args.chunk_size,
        output_jsonl=Path(args.output_jsonl),
        output_schema=Path(args.output_schema),
        output_chunks_dir=Path(args.output_chunks_dir),
    )


if __name__ == "__main__":
    main()
