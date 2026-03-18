from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

from maner.core.types import Sample


class DatasetReader(ABC):
    @abstractmethod
    def iter_samples(self) -> Iterator[Sample]:
        raise NotImplementedError


class GenericJSONLReader(DatasetReader):
    def __init__(self, data_path: str | Path):
        self.data_path = Path(data_path)

    def iter_samples(self) -> Iterator[Sample]:
        with self.data_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    obj = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {self.data_path}:{line_no}: {exc}") from exc

                sample_id = str(obj.get("id", "")).strip()
                body = obj.get("text", None)
                if not sample_id:
                    raise ValueError(f"Missing 'id' at {self.data_path}:{line_no}")
                if not isinstance(body, str):
                    raise ValueError(f"Missing or invalid 'text' at {self.data_path}:{line_no}")

                gold_mentions = obj.get("gold_mentions")
                if gold_mentions is not None and not isinstance(gold_mentions, list):
                    raise ValueError(f"'gold_mentions' must be list at {self.data_path}:{line_no}")
                yield Sample(sample_id=sample_id, text=body, gold_mentions=gold_mentions)


def build_reader(data_path: str | Path, reader_type: str = "generic_jsonl") -> DatasetReader:
    if reader_type == "generic_jsonl":
        return GenericJSONLReader(data_path=data_path)
    raise ValueError(f"Unsupported reader_type: {reader_type}")
