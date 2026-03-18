from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class MemoryItem:
    kind: str
    key: str
    value: dict[str, Any]
    confidence: float
    status: str
    seen_count: int


class MemoryStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        self.conn.close()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL,
                item_key TEXT NOT NULL,
                value_json TEXT NOT NULL,
                confidence REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'candidate',
                seen_count INTEGER NOT NULL DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_kind_key ON memory_items(kind, item_key)"
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_status ON memory_items(status)")
        self.conn.commit()

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        kinds: list[str] | None = None,
        include_candidate: bool = True,
        min_seen_count: int = 1,
        min_confidence: float = 0.0,
    ) -> list[dict[str, Any]]:
        where = "status IN ('active', 'candidate')" if include_candidate else "status = 'active'"
        params: list[Any] = []
        where += " AND seen_count >= ?"
        params.append(max(1, int(min_seen_count)))
        where += " AND confidence >= ?"
        params.append(max(0.0, float(min_confidence)))
        if kinds:
            ph = ",".join("?" for _ in kinds)
            where += f" AND kind IN ({ph})"
            params.extend(kinds)

        rows = self.conn.execute(
            f"SELECT kind, item_key, value_json, confidence, status, seen_count FROM memory_items WHERE {where}",
            params,
        ).fetchall()

        tokens = _tokenize(query)
        scored: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            text = f"{row['kind']} {row['item_key']} {row['value_json']}".lower()
            score = float(row["confidence"])
            if tokens:
                score += sum(1.0 for t in tokens if t in text)
            scored.append((score, row))

        scored.sort(
            key=lambda x: (
                1 if x[1]["status"] == "active" else 0,
                x[0],
                int(x[1]["seen_count"]),
            ),
            reverse=True,
        )

        result: list[dict[str, Any]] = []
        for score, row in scored[:top_k]:
            result.append(
                {
                    "kind": row["kind"],
                    "key": row["item_key"],
                    "value": json.loads(row["value_json"]),
                    "confidence": float(row["confidence"]),
                    "status": row["status"],
                    "seen_count": int(row["seen_count"]),
                    "score": score,
                }
            )
        return result

    def writeback(
        self,
        kind: str,
        key: str,
        value: dict[str, Any],
        confidence: float,
        verifier_pass: bool,
        promote_threshold: int = 2,
        min_confidence: float = 0.7,
    ) -> dict[str, Any]:
        if not verifier_pass:
            return {"written": False, "reason": "verifier_fail"}
        if confidence < min_confidence:
            return {"written": False, "reason": "low_confidence"}

        value_json = json.dumps(value, ensure_ascii=False, sort_keys=True)
        row = self.conn.execute(
            """
            SELECT id, seen_count, status
            FROM memory_items
            WHERE kind = ? AND item_key = ? AND value_json = ?
            """,
            (kind, key, value_json),
        ).fetchone()

        if row is None:
            seen_count = 1
            status = "candidate"
            self.conn.execute(
                """
                INSERT INTO memory_items(kind, item_key, value_json, confidence, status, seen_count)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (kind, key, value_json, confidence, status, seen_count),
            )
        else:
            seen_count = int(row["seen_count"]) + 1
            status = row["status"]
            self.conn.execute(
                """
                UPDATE memory_items
                SET seen_count = ?, confidence = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (seen_count, max(confidence, 0.0), int(row["id"])),
            )

        if seen_count >= promote_threshold:
            status = "active"
            self.conn.execute(
                """
                UPDATE memory_items
                SET status = 'active', updated_at = CURRENT_TIMESTAMP
                WHERE kind = ? AND item_key = ? AND value_json = ?
                """,
                (kind, key, value_json),
            )

        self.conn.commit()
        return {"written": True, "status": status, "seen_count": seen_count}


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in text.replace("_", " ").split() if len(t) >= 2]
