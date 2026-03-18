from __future__ import annotations

from maner.core.types import ConflictCluster, Mention


def _max_confidence(mentions: list[Mention]) -> float:
    if not mentions:
        return 0.0
    return max(m.confidence for m in mentions)


def score_conflict_cluster(
    cluster: ConflictCluster,
    exp_mentions: list[Mention],
    re_mentions: list[Mention],
    relation_constraint_strength: float = 0.0,
) -> float:
    score = 0.0
    if "type" in cluster.conflicts:
        score += 0.45
    if "boundary" in cluster.conflicts:
        score += 0.30
    if "existence" in cluster.conflicts:
        score += 0.35

    exp_conf = _max_confidence(exp_mentions)
    re_conf = _max_confidence(re_mentions)
    if exp_mentions and re_mentions and abs(exp_conf - re_conf) < 0.15:
        score += 0.2

    score += max(0.0, min(relation_constraint_strength, 0.2))
    return min(score, 1.0)


def assign_risk_level(score: float, l2_threshold: float = 0.4, l3_threshold: float = 0.75) -> str:
    if score >= l3_threshold:
        return "L3"
    if score >= l2_threshold:
        return "L2"
    return "L1"


def triage_conflicts(
    clusters: list[ConflictCluster],
    y_exp_mentions: list[Mention],
    y_re_mentions: list[Mention],
    l2_threshold: float = 0.4,
    l3_threshold: float = 0.75,
    relation_constraint_strength: float = 0.0,
) -> dict[str, tuple[float, str]]:
    exp_by_span = {m.span_id: m for m in y_exp_mentions}
    re_by_span = {m.span_id: m for m in y_re_mentions}

    scored: dict[str, tuple[float, str]] = {}
    for c in clusters:
        exp_mentions = [exp_by_span[sid] for sid in c.span_ids if sid in exp_by_span]
        re_mentions = [re_by_span[sid] for sid in c.span_ids if sid in re_by_span]
        score = score_conflict_cluster(
            cluster=c,
            exp_mentions=exp_mentions,
            re_mentions=re_mentions,
            relation_constraint_strength=relation_constraint_strength,
        )
        level = assign_risk_level(score, l2_threshold=l2_threshold, l3_threshold=l3_threshold)
        scored[c.cluster_id] = (score, level)
    return scored
