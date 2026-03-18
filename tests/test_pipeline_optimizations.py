import json

from maner.core.config import load_yaml
from maner.core.types import CandidateSet, Mention, Span, UsageCost
from maner.orchestrator.pipeline import (
    PipelineRunner,
    _drop_mentions_with_negative_rationale,
    _drop_blocklisted_mentions,
    _inject_configured_candidate_mentions,
    _normalize_candidate_span_boundaries,
    _postprocess_final_mentions,
)


def test_boundary_normalization_extends_left_modifier() -> None:
    text = "human gp330 is expressed."
    start = text.index("gp330")
    end = start + len("gp330")
    cset = CandidateSet(has_entity=True, spans={"sp_0001": Span(text="gp330", start=start, end=end)})

    updated, trace = _normalize_candidate_span_boundaries(
        candidate_set=cset,
        text=text,
        left_modifiers=["human"],
        right_modifiers=[],
        enable_hyphen_left=False,
    )

    assert updated == ["sp_0001"]
    assert trace["updated_count"] == 1
    assert cset.spans["sp_0001"].text == "human gp330"


def test_drop_mentions_with_negative_rationale() -> None:
    mentions = [
        Mention(
            span_id="sp_0001",
            span=Span(text="Abl", start=0, end=3),
            ent_type="ENTITY",
            confidence=0.9,
            evidence=[],
            rationale="is an entity",
        ),
        Mention(
            span_id="sp_0002",
            span=Span(text="study", start=4, end=9),
            ent_type="ENTITY",
            confidence=0.9,
            evidence=[],
            rationale="not a named entity mention",
        ),
        Mention(
            span_id="sp_0003",
            span=Span(text="WHS", start=10, end=13),
            ent_type="ENTITY",
            confidence=0.9,
            evidence=[],
            rationale="excluded per expert constraints",
        ),
    ]

    kept, dropped = _drop_mentions_with_negative_rationale(mentions)
    assert dropped == 0
    assert len(kept) == 3


def test_pipeline_short_circuit_when_no_candidate_after_augmentation(tmp_path, monkeypatch) -> None:
    cfg = load_yaml("configs/default.yaml")
    cfg["output"] = {"predictions_path": str(tmp_path / "pred.short.jsonl")}
    cfg["memory"]["sqlite_path"] = str(tmp_path / "memory.short.db")
    cfg["pipeline"]["allow_expert_span_augmentation"] = False
    cfg["pipeline"]["allow_re_span_augmentation"] = False

    runner = PipelineRunner(cfg)
    try:
        def _fake_candidate_run(text, schema):
            return CandidateSet(has_entity=False, spans={}), UsageCost(), {"agent": "candidate"}

        monkeypatch.setattr(runner.candidate_agent, "run", _fake_candidate_run)
        records = runner.run()
    finally:
        runner.close()

    assert records
    rec = records[0]
    assert rec["mentions"] == []
    assert rec["traces"]["short_circuit"]["enabled"] is True
    assert rec["traces"]["ner_expert"]["skipped"] is True
    assert rec["traces"]["ner_re"]["skipped"] is True


def test_postprocess_drops_subsumed_shorter_mention() -> None:
    text = "Raf - 1 activity increased."
    mentions = [
        Mention(
            span_id="sp_0001",
            span=Span(text="Raf - 1", start=0, end=7),
            ent_type="GENE",
            confidence=0.95,
            evidence=[],
            rationale="long",
        ),
        Mention(
            span_id="sp_0002",
            span=Span(text="Raf", start=0, end=3),
            ent_type="GENE",
            confidence=0.90,
            evidence=[],
            rationale="short",
        ),
    ]

    out, trace = _postprocess_final_mentions(text, mentions)
    assert (
        trace["subsumed_dropped"] >= 1
        or trace["offset_type_dedup_dropped"] >= 1
        or trace["hyphen_expanded"] >= 1
    )
    assert len(out) == 1
    assert out[0].span.text == "Raf - 1"


def test_postprocess_merges_adjacent_hyphen_mentions() -> None:
    text = "M1Ach - muscarinic receptor is involved."
    mentions = [
        Mention(
            span_id="sp_0001",
            span=Span(text="M1Ach", start=0, end=5),
            ent_type="GENE",
            confidence=0.90,
            evidence=[],
            rationale="left",
        ),
        Mention(
            span_id="sp_0002",
            span=Span(text="muscarinic receptor", start=8, end=27),
            ent_type="GENE",
            confidence=0.90,
            evidence=[],
            rationale="right",
        ),
    ]

    out, trace = _postprocess_final_mentions(text, mentions)
    assert trace["adjacent_merged"] >= 1 or trace["hyphen_expanded"] >= 1
    assert len(out) == 1
    assert out[0].span.text == "M1Ach - muscarinic receptor"


def test_postprocess_expands_single_hyphenated_suffix() -> None:
    text = "Raf - 1 activity increased."
    mentions = [
        Mention(
            span_id="sp_0001",
            span=Span(text="Raf", start=0, end=3),
            ent_type="GENE",
            confidence=0.9,
            evidence=[],
            rationale="base",
        )
    ]

    out, trace = _postprocess_final_mentions(text, mentions)
    assert trace["hyphen_expanded"] >= 1
    assert len(out) == 1
    assert out[0].span.text == "Raf - 1"


def test_postprocess_does_not_merge_slash_mentions() -> None:
    text = "AD / SDAT biomarkers"
    mentions = [
        Mention(
            span_id="sp_0001",
            span=Span(text="AD", start=0, end=2),
            ent_type="GENE",
            confidence=0.9,
            evidence=[],
            rationale="left",
        ),
        Mention(
            span_id="sp_0002",
            span=Span(text="SDAT", start=5, end=9),
            ent_type="GENE",
            confidence=0.9,
            evidence=[],
            rationale="right",
        ),
    ]

    out, trace = _postprocess_final_mentions(text, mentions)
    assert trace["adjacent_merged"] == 0
    assert len(out) == 2


def test_postprocess_splits_slash_coordinated_entity() -> None:
    text = "MMAC1 / PTEN inhibits growth."
    mentions = [
        Mention(
            span_id="sp_0001",
            span=Span(text="MMAC1 / PTEN", start=0, end=12),
            ent_type="GENE",
            confidence=0.95,
            evidence=[],
            rationale="combined",
        )
    ]

    out, trace = _postprocess_final_mentions(text, mentions)
    spans = sorted((m.span.start, m.span.end, m.span.text) for m in out)
    assert trace["slash_split"] >= 1
    assert (0, 5, "MMAC1") in spans
    assert (8, 12, "PTEN") in spans


def test_postprocess_descriptor_expands_attached_suffix_and_mutant() -> None:
    text = "cdc42 - 1ts mutant shows phenotype."
    mentions = [
        Mention(
            span_id="sp_0001",
            span=Span(text="cdc42 - 1", start=0, end=9),
            ent_type="GENE",
            confidence=0.9,
            evidence=[],
            rationale="base",
        )
    ]

    out, trace = _postprocess_final_mentions(
        text=text,
        mentions=mentions,
        enable_descriptor_expansion=True,
        descriptor_terms=["mutant", "mutants"],
        descriptor_left_modifiers=[],
    )
    assert trace["descriptor_expanded"] >= 1
    assert any(m.span.text == "cdc42 - 1ts mutant" for m in out)


def test_postprocess_descriptor_continues_after_existing_descriptor_token() -> None:
    text = "service platform 1 support service is detected."
    mentions = [
        Mention(
            span_id="sp_0001",
            span=Span(text="service platform 1", start=0, end=18),
            ent_type="GENE",
            confidence=0.9,
            evidence=[],
            rationale="partial",
        )
    ]

    out, trace = _postprocess_final_mentions(
        text=text,
        mentions=mentions,
        enable_descriptor_expansion=True,
        descriptor_terms=["service", "platform", "support"],
        descriptor_left_modifiers=[],
    )
    assert trace["descriptor_expanded"] >= 1
    assert any(m.span.text == "service platform 1 support service" for m in out)


def test_postprocess_descriptor_can_expand_lowercase_prefix() -> None:
    text = "adenovirus E1B mutant induces apoptosis."
    mentions = [
        Mention(
            span_id="sp_0001",
            span=Span(text="E1B", start=11, end=14),
            ent_type="GENE",
            confidence=0.9,
            evidence=[],
            rationale="base",
        )
    ]

    out, trace = _postprocess_final_mentions(
        text=text,
        mentions=mentions,
        enable_descriptor_expansion=True,
        descriptor_terms=["mutant", "mutants"],
        descriptor_left_modifiers=[],
    )
    assert trace["descriptor_expanded"] >= 1
    assert any(m.span.text == "adenovirus E1B mutant" for m in out)


def test_configured_lexical_injection_uses_rules_only() -> None:
    text = "Acme Corp hired John."
    cset = CandidateSet(
        has_entity=True,
        spans={
            "sp_0001": Span(text="Acme Corp", start=0, end=9),
            "sp_0002": Span(text="John", start=16, end=20),
        },
    )
    mentions: list[Mention] = []

    out, trace = _inject_configured_candidate_mentions(
        text=text,
        candidate_set=cset,
        mentions=mentions,
        rules=[{"term": "Acme Corp", "ent_type": "ORG", "confidence": 0.8}],
        default_ent_type="ENTITY",
    )

    assert trace["enabled"] is True
    assert trace["added"] == 1
    assert len(out) == 1
    assert out[0].ent_type == "ORG"


def test_blocklist_filter_is_disabled_without_terms() -> None:
    mentions = [
        Mention(
            span_id="sp_0001",
            span=Span(text="Acme", start=0, end=4),
            ent_type="ORG",
            confidence=0.9,
            evidence=[],
            rationale="",
        )
    ]

    out, trace = _drop_blocklisted_mentions(mentions, blocked_terms=set())
    assert trace["enabled"] is False
    assert trace["dropped"] == 0
    assert len(out) == 1


def test_pipeline_retries_retryable_sample_error(tmp_path, monkeypatch) -> None:
    data_path = tmp_path / "one_sample.jsonl"
    data_path.write_text(
        json.dumps({"id": "s1", "text": "Abl activates pathway."}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    cfg = load_yaml("configs/default.yaml")
    cfg["data"]["data_path"] = str(data_path)
    cfg["output"] = {"predictions_path": str(tmp_path / "pred.retry.jsonl")}
    cfg["memory"]["sqlite_path"] = str(tmp_path / "memory.retry.db")
    cfg["pipeline"]["sample_max_retries"] = 1
    cfg["pipeline"]["sample_retry_backoff_s"] = 0.0

    runner = PipelineRunner(cfg)
    calls = {"n": 0}
    try:
        def _fake_run_sample(sample_id: str, text: str):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("LLM network error: simulated disconnect")
            return {
                "id": sample_id,
                "text": text,
                "mentions": [],
                "traces": {"ok": True},
                "costs": {
                    "calls": 0,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                    "latency_ms": [],
                    "debate_turns": 0,
                    "debate_triggered": 0,
                    "debate_trigger_rate": 0.0,
                },
            }

        monkeypatch.setattr(runner, "_run_sample", _fake_run_sample)
        records = runner.run()
    finally:
        runner.close()

    assert len(records) == 1
    assert "runtime_error" not in (records[0].get("traces") or {})
    assert calls["n"] == 2


def test_pipeline_post_error_retry_recovers_sample(tmp_path, monkeypatch) -> None:
    data_path = tmp_path / "one_sample_post_retry.jsonl"
    data_path.write_text(
        json.dumps({"id": "s1", "text": "PTEN regulates pathway."}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    cfg = load_yaml("configs/default.yaml")
    cfg["data"]["data_path"] = str(data_path)
    cfg["output"] = {"predictions_path": str(tmp_path / "pred.postretry.jsonl")}
    cfg["memory"]["sqlite_path"] = str(tmp_path / "memory.postretry.db")
    cfg["pipeline"]["sample_max_retries"] = 0
    cfg["pipeline"]["sample_retry_backoff_s"] = 0.0
    cfg["pipeline"]["sample_post_error_retries"] = 1
    cfg["pipeline"]["sample_post_error_backoff_s"] = 0.0

    runner = PipelineRunner(cfg)
    calls = {"n": 0}
    try:
        def _fake_run_sample(sample_id: str, text: str):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("LLM network error: transient")
            return {
                "id": sample_id,
                "text": text,
                "mentions": [],
                "traces": {"ok": True},
                "costs": {
                    "calls": 0,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                    "latency_ms": [],
                    "debate_turns": 0,
                    "debate_triggered": 0,
                    "debate_trigger_rate": 0.0,
                },
            }

        monkeypatch.setattr(runner, "_run_sample", _fake_run_sample)
        records = runner.run()
    finally:
        runner.close()

    assert len(records) == 1
    assert "runtime_error" not in (records[0].get("traces") or {})
    assert calls["n"] == 2
