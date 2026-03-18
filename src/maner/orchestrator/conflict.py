from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from maner.core.types import ConflictCluster, Mention, NERHypothesis, span_iou


@dataclass
class MentionPair:
    exp: Mention | None
    re: Mention | None


def _best_iou_match(
    target: Mention, candidates: list[Mention], used: set[int], threshold: float
) -> int | None:
    best_idx = None
    best_score = 0.0
    for idx, cand in enumerate(candidates):
        if idx in used:
            continue
        score = span_iou(target.span, cand.span)
        if score >= threshold and score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def build_conflict_clusters(
    y_exp: NERHypothesis,
    y_re: NERHypothesis,
    iou_threshold: float = 0.5,
) -> tuple[list[ConflictCluster], dict[str, Any]]:
    exp_mentions = list(y_exp.mentions)
    re_mentions = list(y_re.mentions)

    clusters: list[ConflictCluster] = []
    alignment: list[MentionPair] = []

    re_by_span_id = {m.span_id: m for m in re_mentions}
    matched_re_ids: set[str] = set()

    for m_exp in exp_mentions:
        if m_exp.span_id in re_by_span_id:
            m_re = re_by_span_id[m_exp.span_id]
            matched_re_ids.add(m_re.span_id)
            alignment.append(MentionPair(exp=m_exp, re=m_re))
        else:
            alignment.append(MentionPair(exp=m_exp, re=None))

    remaining_re = [m for m in re_mentions if m.span_id not in matched_re_ids]
    used_re_indices: set[int] = set()

    for idx, pair in enumerate(alignment):
        if pair.re is not None or pair.exp is None:
            continue
        best_idx = _best_iou_match(pair.exp, remaining_re, used_re_indices, iou_threshold)
        if best_idx is not None:
            alignment[idx] = MentionPair(exp=pair.exp, re=remaining_re[best_idx])
            used_re_indices.add(best_idx)

    for idx, m_re in enumerate(remaining_re):
        if idx not in used_re_indices:
            alignment.append(MentionPair(exp=None, re=m_re))

    cluster_idx = 1
    for pair in alignment:
        conflicts: list[str] = []
        span_ids: list[str] = []

        if pair.exp is not None:
            span_ids.append(pair.exp.span_id)
        if pair.re is not None and pair.re.span_id not in span_ids:
            span_ids.append(pair.re.span_id)

        if pair.exp is None or pair.re is None:
            conflicts.append("existence")
        else:
            if pair.exp.ent_type != pair.re.ent_type:
                conflicts.append("type")
            if pair.exp.span.start != pair.re.span.start or pair.exp.span.end != pair.re.span.end:
                conflicts.append("boundary")

        if not conflicts:
            continue

        clusters.append(
            ConflictCluster(
                cluster_id=f"cluster_{cluster_idx:04d}",
                span_ids=span_ids,
                conflicts=conflicts,
                risk_level="L1",
                score=0.0,
            )
        )
        cluster_idx += 1

    trace = {
        "alignment_pairs": [
            {
                "exp": pair.exp.span_id if pair.exp else None,
                "re": pair.re.span_id if pair.re else None,
            }
            for pair in alignment
        ]
    }
    return clusters, trace


def apply_risk_levels(
    clusters: list[ConflictCluster], scored: dict[str, tuple[float, str]]
) -> list[ConflictCluster]:
    updated: list[ConflictCluster] = []
    for c in clusters:
        score, level = scored.get(c.cluster_id, (c.score, c.risk_level))
        updated.append(replace(c, score=score, risk_level=level))
    return updated
