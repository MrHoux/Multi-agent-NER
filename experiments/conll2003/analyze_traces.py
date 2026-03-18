from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", required=True)
    args = parser.parse_args()

    rows = _load_jsonl(Path(args.pred_path))
    conflict = 0
    disamb_drop = 0
    re_filter_drop = 0
    memory_items = 0
    type_counter: Counter[str] = Counter()

    for r in rows:
        for m in r.get("mentions", []) or []:
            type_counter[str(m.get("ent_type", "UNK"))] += 1
        tr = r.get("traces", {}) or {}
        clusters = (tr.get("conflict", {}) or {}).get("clusters", [])
        if isinstance(clusters, list):
            conflict += len(clusters)
        dis = tr.get("disambiguation_agent", {}) or {}
        disamb_drop += int(dis.get("dropped", 0) or 0)
        rf = tr.get("re_collab_filter", {}) or {}
        re_filter_drop += int(rf.get("dropped_low_conf", 0) or 0)
        re_filter_drop += int(rf.get("dropped_no_evidence", 0) or 0)
        re_filter_drop += int(rf.get("dropped_expert_override", 0) or 0)
        re_filter_drop += int(rf.get("dropped_re_only_cap", 0) or 0)
        mr = tr.get("memory_retrieve", [])
        if isinstance(mr, list):
            memory_items += len(mr)

    print(json.dumps(
        {
            "samples": len(rows),
            "final_mentions": sum(len(r.get("mentions", []) or []) for r in rows),
            "sum_conflict_clusters": conflict,
            "sum_disamb_drop": disamb_drop,
            "sum_re_filter_drop": re_filter_drop,
            "sum_memory_retrieved_items": memory_items,
            "final_type_counts": dict(type_counter),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
