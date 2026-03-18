from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass
class Span:
    text: str
    start: int
    end: int
    provenance: dict[str, Any] | None = None


@dataclass
class Evidence:
    quote: str
    start: int
    end: int


@dataclass
class Mention:
    span_id: str
    span: Span
    ent_type: str
    confidence: float
    evidence: list[Evidence] = field(default_factory=list)
    rationale: str = ""


@dataclass
class CandidateSet:
    has_entity: bool
    spans: dict[str, Span] = field(default_factory=dict)


@dataclass
class BoundaryOp:
    op: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpanConstraint:
    candidate_types: list[str] = field(default_factory=list)
    excluded_types: list[str] = field(default_factory=list)
    boundary_ops: list[BoundaryOp] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class ExpertConstraints:
    terminology: list[str] = field(default_factory=list)
    triggers: list[str] = field(default_factory=list)
    per_span: dict[str, SpanConstraint] = field(default_factory=dict)


@dataclass
class Relation:
    head_span_id: str
    rel_type: str
    tail_span_id: str
    confidence: float
    evidence: list[Evidence] = field(default_factory=list)


@dataclass
class NERHypothesis:
    mentions: list[Mention]
    source: Literal["expert", "re"]


@dataclass
class ConflictCluster:
    cluster_id: str
    span_ids: list[str]
    conflicts: list[str]
    risk_level: Literal["L1", "L2", "L3"]
    score: float


@dataclass
class Decision:
    final_mentions: list[Mention]
    trace: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class UsageCost:
    calls: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: list[int] = field(default_factory=list)
    debate_turns: int = 0
    debate_triggered: int = 0


@dataclass
class Sample:
    sample_id: str
    text: str
    gold_mentions: list[dict[str, Any]] | None = None


def to_dict(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, list):
        return [to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    return obj


def span_overlap(a: Span, b: Span) -> int:
    return max(0, min(a.end, b.end) - max(a.start, b.start))


def span_iou(a: Span, b: Span) -> float:
    overlap = span_overlap(a, b)
    if overlap == 0:
        return 0.0
    union = (a.end - a.start) + (b.end - b.start) - overlap
    if union <= 0:
        return 0.0
    return overlap / union


def merge_spans(spans: list[Span]) -> list[Span]:
    if not spans:
        return []
    sorted_spans = sorted(spans, key=lambda s: (s.start, s.end))
    merged: list[Span] = [sorted_spans[0]]
    for span in sorted_spans[1:]:
        last = merged[-1]
        if span.start <= last.end:
            merged[-1] = Span(
                text=last.text,
                start=last.start,
                end=max(last.end, span.end),
                provenance={"merged_from": [to_dict(last), to_dict(span)]},
            )
        else:
            merged.append(span)
    return merged


def is_valid_offsets(text: str, start: int, end: int) -> bool:
    return 0 <= start <= end <= len(text)


def is_strict_substring(text: str, quote: str, start: int, end: int) -> bool:
    if not is_valid_offsets(text, start, end):
        return False
    return text[start:end] == quote


def apply_boundary_op(base_span_id: str, base_span: Span, op: BoundaryOp, text: str) -> list[Span]:
    if op.op.upper() == "TRIM":
        l = int(op.params.get("left", 0))
        r = int(op.params.get("right", 0))
        start = max(base_span.start + l, base_span.start)
        end = min(base_span.end - r, base_span.end)
        if not is_valid_offsets(text, start, end) or start >= end:
            return [base_span]
        return [
            Span(
                text=text[start:end],
                start=start,
                end=end,
                provenance={"source_span_id": base_span_id, "op": "TRIM", "params": op.params},
            )
        ]
    if op.op.upper() == "SPLIT":
        cut = int(op.params.get("cut", base_span.start + (base_span.end - base_span.start) // 2))
        if cut <= base_span.start or cut >= base_span.end:
            return [base_span]
        return [
            Span(
                text=text[base_span.start : cut],
                start=base_span.start,
                end=cut,
                provenance={"source_span_id": base_span_id, "op": "SPLIT", "part": 0},
            ),
            Span(
                text=text[cut : base_span.end],
                start=cut,
                end=base_span.end,
                provenance={"source_span_id": base_span_id, "op": "SPLIT", "part": 1},
            ),
        ]
    return [base_span]
