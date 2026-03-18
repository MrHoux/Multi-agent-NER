from maner.agents.expert_agent import ExpertAgent
from maner.agents.re_agent import REAgent
from maner.core.prompting import PromptManager
from maner.core.types import CandidateSet, Span
from maner.llm.client import LLMClient
from maner.orchestrator.pipeline import _augment_candidate_set


def test_augment_candidate_set_adds_only_new_valid_spans() -> None:
    text = "alpha beta gamma"
    cset = CandidateSet(
        has_entity=True,
        spans={
            "sp_0001": Span(text="alpha", start=0, end=5),
        },
    )
    proposals = [
        {"source": "expert", "start": 0, "end": 5, "text": "alpha"},
        {"source": "expert", "start": 6, "end": 10, "text": "beta", "rationale": "new"},
        {"source": "re", "start": -1, "end": 2, "text": "xx"},
    ]

    added, trace = _augment_candidate_set(cset, text, proposals, max_new_spans=10)

    assert len(added) == 1
    assert trace["added_count"] == 1
    sid = added[0]
    assert sid in cset.spans
    assert cset.spans[sid].text == "beta"
    assert cset.spans[sid].provenance is not None
    assert cset.spans[sid].provenance["source_agent"] == "expert"


def test_expert_parse_span_proposals_filters_invalid() -> None:
    text = "John works at Acme Corp."
    agent = ExpertAgent(LLMClient({"provider": "mock"}), PromptManager("configs/prompts_cot.yaml"))
    payload = {
        "new_spans": [
            {
                "start": 14,
                "end": 23,
                "text": "Acme Corp",
                "confidence": 0.8,
                "evidence": [{"quote": "Acme Corp", "start": 14, "end": 23}],
            },
            {"start": 50, "end": 60, "text": "bad"},
        ]
    }

    out = agent._parse_span_proposals(payload, text, source="expert")
    assert len(out) == 1
    assert out[0]["text"] == "Acme Corp"
    assert out[0]["source"] == "expert"


def test_expert_parse_span_proposals_enforces_schema_type_hints() -> None:
    text = "Abl regulates trio."
    agent = ExpertAgent(LLMClient({"provider": "mock"}), PromptManager("configs/prompts_cot.yaml"))
    payload = {
        "new_spans": [
            {
                "start": 0,
                "end": 3,
                "text": "Abl",
                "type_hints": ["GENE"],
                "confidence": 0.9,
                "evidence": [{"quote": "Abl", "start": 0, "end": 3}],
            },
            {
                "start": 14,
                "end": 18,
                "text": "trio",
                "type_hints": ["DISEASE"],
                "confidence": 0.9,
                "evidence": [{"quote": "trio", "start": 14, "end": 18}],
            },
        ]
    }

    out = agent._parse_span_proposals(payload, text, source="expert", valid_types={"GENE"})
    assert len(out) == 1
    assert out[0]["text"] == "Abl"
    assert out[0]["type_hints"] == ["GENE"]


def test_re_parse_span_proposals_filters_invalid() -> None:
    text = "Abl regulates trio."
    agent = REAgent(LLMClient({"provider": "mock"}), PromptManager("configs/prompts_cot.yaml"))
    payload = {
        "new_spans": [
            {
                "start": 0,
                "end": 3,
                "text": "Abl",
                "confidence": 0.9,
                "evidence": [{"quote": "Abl", "start": 0, "end": 3}],
            },
            {"start": 3, "end": 3, "text": ""},
        ]
    }

    out = agent._parse_span_proposals(payload, text, source="re")
    assert len(out) == 1
    assert out[0]["text"] == "Abl"
    assert out[0]["source"] == "re"


def test_re_parse_span_proposals_enforces_schema_type_hints() -> None:
    text = "Abl regulates trio."
    agent = REAgent(LLMClient({"provider": "mock"}), PromptManager("configs/prompts_cot.yaml"))
    payload = {
        "new_spans": [
            {
                "start": 0,
                "end": 3,
                "text": "Abl",
                "type_hints": ["GENE"],
                "confidence": 0.9,
                "evidence": [{"quote": "Abl", "start": 0, "end": 3}],
            },
            {
                "start": 14,
                "end": 18,
                "text": "trio",
                "type_hints": ["CONDITION"],
                "confidence": 0.9,
                "evidence": [{"quote": "trio", "start": 14, "end": 18}],
            },
        ]
    }

    out = agent._parse_span_proposals(payload, text, source="re", valid_types={"GENE"})
    assert len(out) == 1
    assert out[0]["text"] == "Abl"
    assert out[0]["type_hints"] == ["GENE"]
