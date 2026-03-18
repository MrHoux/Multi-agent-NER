from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from maner.core.schema import SchemaDefinition


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    def f1(self) -> float:
        p = self.precision()
        r = self.recall()
        return 2 * p * r / (p + r) if (p + r) else 0.0


def _extract_mentions(items: list[dict[str, Any]]) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []
    for m in items:
        ent_type = str(m.get("ent_type", ""))

        if "span" in m and isinstance(m["span"], dict):
            start = int(m["span"].get("start", -1))
            end = int(m["span"].get("end", -1))
        else:
            start = int(m.get("start", -1))
            end = int(m.get("end", -1))

        if start >= 0 and end >= 0 and end >= start and ent_type:
            out.append((start, end, ent_type))
    return out


def _load_gold(path: str | Path) -> dict[str, list[tuple[int, int, str]]]:
    records: dict[str, list[tuple[int, int, str]]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            sid = str(obj.get("id", ""))
            mentions = _extract_mentions(obj.get("gold_mentions", []) or [])
            records[sid] = mentions
    return records


def _load_pred(path: str | Path) -> dict[str, list[tuple[int, int, str]]]:
    records: dict[str, list[tuple[int, int, str]]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            sid = str(obj.get("id", ""))
            mentions = _extract_mentions(obj.get("mentions", []) or [])
            records[sid] = mentions
    return records


def _span_overlap(a: tuple[int, int, str], b: tuple[int, int, str]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def compute_strict_span_f1(
    gold_records: dict[str, list[tuple[int, int, str]]],
    pred_records: dict[str, list[tuple[int, int, str]]],
    entity_types: list[str],
) -> dict[str, Any]:
    per_type = {t: Counts() for t in entity_types}
    micro = Counts()

    all_ids = set(gold_records) | set(pred_records)
    for sid in all_ids:
        gold_set = set(gold_records.get(sid, []))
        pred_set = set(pred_records.get(sid, []))

        tp = gold_set & pred_set
        fp = pred_set - gold_set
        fn = gold_set - pred_set

        micro.tp += len(tp)
        micro.fp += len(fp)
        micro.fn += len(fn)

        for _, _, t in tp:
            per_type.setdefault(t, Counts()).tp += 1
        for _, _, t in fp:
            per_type.setdefault(t, Counts()).fp += 1
        for _, _, t in fn:
            per_type.setdefault(t, Counts()).fn += 1

    macro_f1_vals = [c.f1() for c in per_type.values()] if per_type else [0.0]
    macro_f1 = sum(macro_f1_vals) / len(macro_f1_vals)

    return {
        "micro": {
            "precision": micro.precision(),
            "recall": micro.recall(),
            "f1": micro.f1(),
            "tp": micro.tp,
            "fp": micro.fp,
            "fn": micro.fn,
        },
        "macro": {"f1": macro_f1},
        "per_type": {
            t: {
                "precision": c.precision(),
                "recall": c.recall(),
                "f1": c.f1(),
                "tp": c.tp,
                "fp": c.fp,
                "fn": c.fn,
            }
            for t, c in per_type.items()
        },
    }


def compute_error_stats(
    gold_records: dict[str, list[tuple[int, int, str]]],
    pred_records: dict[str, list[tuple[int, int, str]]],
) -> dict[str, int]:
    stats = {
        "type_mismatch": 0,
        "boundary_mismatch": 0,
        "spurious": 0,
        "missing": 0,
    }

    all_ids = set(gold_records) | set(pred_records)
    for sid in all_ids:
        gold = gold_records.get(sid, [])
        pred = pred_records.get(sid, [])
        gold_set = set(gold)
        pred_set = set(pred)

        for p in pred:
            if p in gold_set:
                continue
            overlaps = [g for g in gold if _span_overlap(p, g) > 0]
            if not overlaps:
                stats["spurious"] += 1
                continue
            if any((p[0], p[1]) == (g[0], g[1]) and p[2] != g[2] for g in overlaps):
                stats["type_mismatch"] += 1
            else:
                stats["boundary_mismatch"] += 1

        for g in gold:
            if g in pred_set:
                continue
            overlaps = [p for p in pred if _span_overlap(p, g) > 0]
            if not overlaps:
                stats["missing"] += 1

    return stats


def evaluate_from_files(gold_path: str, pred_path: str, schema: SchemaDefinition) -> dict[str, Any]:
    gold_records = _load_gold(gold_path)
    pred_records = _load_pred(pred_path)
    metrics = compute_strict_span_f1(gold_records, pred_records, schema.entity_type_names)
    metrics["errors"] = compute_error_stats(gold_records, pred_records)
    return metrics
