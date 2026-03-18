from maner.core.types import Evidence, Mention, NERHypothesis, Span
from maner.orchestrator.conflict import apply_risk_levels, build_conflict_clusters
from maner.orchestrator.triage import triage_conflicts


def _m(span_id: str, text: str, start: int, end: int, ent_type: str, conf: float) -> Mention:
    return Mention(
        span_id=span_id,
        span=Span(text=text[start:end], start=start, end=end),
        ent_type=ent_type,
        confidence=conf,
        evidence=[Evidence(quote=text[start:end], start=start, end=end)],
        rationale="test",
    )


def test_conflict_cluster_type_and_boundary_and_existence() -> None:
    text = "John works at Acme Corp in Seattle."

    y_exp = NERHypothesis(
        mentions=[
            _m("sp1", text, 0, 4, "PERSON", 0.8),
            _m("sp2", text, 14, 23, "ORG", 0.7),
            _m("sp3", text, 27, 34, "LOC", 0.6),
        ],
        source="expert",
    )

    y_re = NERHypothesis(
        mentions=[
            _m("sp1", text, 0, 4, "ORG", 0.78),
            _m("sp2_alt", text, 14, 22, "ORG", 0.69),
        ],
        source="re",
    )

    clusters, trace = build_conflict_clusters(y_exp, y_re, iou_threshold=0.5)
    assert clusters
    assert trace["alignment_pairs"]

    kinds = [c.conflicts for c in clusters]
    flat = {k for arr in kinds for k in arr}
    assert "type" in flat
    assert "boundary" in flat
    assert "existence" in flat

    scored = triage_conflicts(
        clusters, y_exp.mentions, y_re.mentions, l2_threshold=0.3, l3_threshold=0.7
    )
    updated = apply_risk_levels(clusters, scored)
    assert all(c.risk_level in {"L1", "L2", "L3"} for c in updated)
