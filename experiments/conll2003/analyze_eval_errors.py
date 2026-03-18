from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _mention_key(m: dict[str, Any]) -> tuple[int, int, str]:
    return int(m["start"]), int(m["end"]), str(m["ent_type"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze strict-span NER errors for one file pair.")
    parser.add_argument("--gold_path", required=True)
    parser.add_argument("--pred_path", required=True)
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()

    gold_rows = _load_jsonl(Path(args.gold_path))
    pred_rows = _load_jsonl(Path(args.pred_path))

    pred_by_id = {str(r["id"]): r for r in pred_rows}

    tp = 0
    fp = 0
    fn = 0
    tp_by_type: Counter[str] = Counter()
    fp_by_type: Counter[str] = Counter()
    fn_by_type: Counter[str] = Counter()
    fp_terms: Counter[tuple[str, str]] = Counter()
    fn_terms: Counter[tuple[str, str]] = Counter()
    conf_tp: list[float] = []
    conf_fp: list[float] = []
    conf_by_type: dict[str, list[float]] = defaultdict(list)

    for g in gold_rows:
        sid = str(g["id"])
        p = pred_by_id.get(sid, {})
        gold_mentions = g.get("gold_mentions", []) or []
        pred_mentions = p.get("mentions", []) or []

        gold_set = {_mention_key(m): m for m in gold_mentions}
        pred_set: dict[tuple[int, int, str], dict[str, Any]] = {}
        for m in pred_mentions:
            span = m.get("span", {})
            if not span:
                continue
            ent_type = str(m.get("ent_type", ""))
            key = (int(span.get("start", -1)), int(span.get("end", -1)), ent_type)
            pred_set[key] = m

        for key, m in pred_set.items():
            if key in gold_set:
                tp += 1
                t = key[2]
                tp_by_type[t] += 1
                conf = float(m.get("confidence", 0.0))
                conf_tp.append(conf)
                conf_by_type[t].append(conf)
            else:
                fp += 1
                t = key[2]
                fp_by_type[t] += 1
                span_text = str(m.get("span", {}).get("text", "")).strip()
                fp_terms[(span_text, t)] += 1
                conf_fp.append(float(m.get("confidence", 0.0)))

        for key, m in gold_set.items():
            if key not in pred_set:
                fn += 1
                t = key[2]
                fn_by_type[t] += 1
                fn_terms[(str(m.get("text", "")).strip(), t)] += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"micro: p={precision:.4f} r={recall:.4f} f1={f1:.4f} tp={tp} fp={fp} fn={fn}")
    print("by_type:")
    all_types = sorted(set(tp_by_type) | set(fp_by_type) | set(fn_by_type))
    for t in all_types:
        print(
            f"  {t}: tp={tp_by_type[t]} fp={fp_by_type[t]} fn={fn_by_type[t]} "
            f"avg_conf_tp={mean(conf_by_type[t]):.3f}" if conf_by_type[t] else f"  {t}: tp={tp_by_type[t]} fp={fp_by_type[t]} fn={fn_by_type[t]} avg_conf_tp=n/a"
        )

    print(f"avg_conf_tp={mean(conf_tp):.4f}" if conf_tp else "avg_conf_tp=n/a")
    print(f"avg_conf_fp={mean(conf_fp):.4f}" if conf_fp else "avg_conf_fp=n/a")

    print(f"top_fp_terms (k={args.top_k}):")
    for (term, t), c in fp_terms.most_common(args.top_k):
        print(f"  {term!r} [{t}] x{c}")

    print(f"top_fn_terms (k={args.top_k}):")
    for (term, t), c in fn_terms.most_common(args.top_k):
        print(f"  {term!r} [{t}] x{c}")


if __name__ == "__main__":
    main()
