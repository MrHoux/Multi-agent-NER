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


def _reconstruct_text_and_offsets(tokens: list[str]) -> tuple[str, list[int], list[int]]:
    starts: list[int] = []
    ends: list[int] = []
    chunks: list[str] = []
    cursor = 0

    for idx, tok in enumerate(tokens):
        if idx > 0:
            chunks.append(" ")
            cursor += 1
        starts.append(cursor)
        chunks.append(tok)
        cursor += len(tok)
        ends.append(cursor)

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
        mention = {
            "start": current_start,
            "end": current_end,
            "ent_type": current_type,
            "text": text[current_start:current_end],
        }
        mentions.append(mention)
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


def prepare_bc2gm(
    dataset_name: str,
    split: str,
    max_samples: int | None,
    output_jsonl: Path,
    output_schema: Path,
) -> tuple[Path, Path]:
    load_dataset = _import_hf_load_dataset()
    ds = load_dataset(dataset_name)
    if split not in ds:
        raise ValueError(f"Split '{split}' not in dataset. Available: {list(ds.keys())}")

    split_ds = ds[split]
    if max_samples is not None:
        split_ds = split_ds.select(range(min(max_samples, len(split_ds))))

    feature = split_ds.features["ner_tags"].feature
    label_names = list(getattr(feature, "names", []))
    if not label_names:
        raise ValueError("Failed to read label names from ner_tags feature.")

    entity_types = sorted(
        {label.split("-", 1)[1] for label in label_names if label != "O" and "-" in label}
    )
    entity_type_label_map: dict[str, list[str]] = {}
    for label in label_names:
        if label == "O" or "-" not in label:
            continue
        _, ent_type = label.split("-", 1)
        entity_type_label_map.setdefault(ent_type, []).append(label)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_schema.parent.mkdir(parents=True, exist_ok=True)

    with output_jsonl.open("w", encoding="utf-8") as f:
        for row in split_ds:
            tokens = [str(t) for t in row["tokens"]]
            tags = list(row["ner_tags"])
            text, starts, ends = _reconstruct_text_and_offsets(tokens)
            gold_mentions = _bio_to_mentions(tags, starts, ends, text, label_names)
            sample = {
                "id": f"{split}-{row['id']}",
                "text": text,
                "gold_mentions": gold_mentions,
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    schema = {
        "entity_types": [
            {
                "name": ent_type,
                "description": ", ".join(sorted(set(entity_type_label_map.get(ent_type, [ent_type])))),
            }
            for ent_type in entity_types
        ],
        "relation_constraints": [],
    }
    with output_schema.open("w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    print(f"Wrote data: {output_jsonl}")
    print(f"Wrote schema: {output_schema}")
    print(f"Samples: {len(split_ds)} | Entity types: {entity_types}")
    return output_jsonl, output_schema


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BC2GM jsonl + schema for MANER pipeline")
    parser.add_argument("--dataset_name", default="spyysalo/bc2gm_corpus")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--output_jsonl", default="datasets/bc2gm/bc2gm_test.jsonl")
    parser.add_argument("--output_schema", default="datasets/bc2gm/schema.bc2gm.json")
    args = parser.parse_args()

    prepare_bc2gm(
        dataset_name=args.dataset_name,
        split=args.split,
        max_samples=args.max_samples,
        output_jsonl=Path(args.output_jsonl),
        output_schema=Path(args.output_schema),
    )


if __name__ == "__main__":
    main()
