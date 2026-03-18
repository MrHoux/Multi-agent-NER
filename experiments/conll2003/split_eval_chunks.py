from __future__ import annotations

import argparse
import json
from pathlib import Path

from maner.core.schema import load_schema
from maner.eval.metrics import evaluate_from_files


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
    parser.add_argument("--chunks_dir", default="datasets/conll2003/chunks100")
    parser.add_argument("--schema_path", default="datasets/conll2003/schema.conll2003.json")
    parser.add_argument("--chunk_count", type=int, default=5)
    parser.add_argument("--out_dir", default="outputs/conll2003/tmp_chunk_split_eval")
    args = parser.parse_args()

    repo = Path(".").resolve()
    pred_path = repo / args.pred_path
    chunks_dir = repo / args.chunks_dir
    schema_path = repo / args.schema_path
    out_dir = repo / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    schema = load_schema(schema_path)
    rows = _load_jsonl(pred_path)
    by_id = {str(r.get("id", "")): r for r in rows}

    result: dict[str, float] = {}
    for i in range(1, args.chunk_count + 1):
        gold_path = chunks_dir / f"conll2003_test.chunk{i:02d}.jsonl"
        gold_rows = _load_jsonl(gold_path)
        pred_chunk_path = out_dir / f"pred.chunk{i:02d}.jsonl"
        with pred_chunk_path.open("w", encoding="utf-8") as f:
            for row in gold_rows:
                sid = str(row.get("id", ""))
                item = by_id.get(sid)
                if item is not None:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        m = evaluate_from_files(str(gold_path), str(pred_chunk_path), schema)
        result[f"chunk{i:02d}"] = float(m["micro"]["f1"])

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
