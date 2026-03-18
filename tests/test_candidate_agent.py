from maner.agents.candidate_agent import CandidateAgent
from maner.core.prompting import PromptManager
from maner.core.schema import load_schema
from maner.llm.client import LLMClient


def test_candidate_agent_high_recall_basic() -> None:
    llm = LLMClient({"provider": "mock"})
    prompts = PromptManager("configs/prompts_cot.yaml")
    schema = load_schema("tests/fixtures/schema_example.json")

    agent = CandidateAgent(llm, prompts)
    cset, cost, _ = agent.run("John works at Acme Corp in Seattle.", schema)

    assert cset.has_entity is True
    assert len(cset.spans) >= 3
    span_texts = [s.text for s in cset.spans.values()]
    assert any("John" in s for s in span_texts)
    assert any("Acme" in s for s in span_texts)
    assert cost.calls == 1


def test_candidate_agent_no_empty_span() -> None:
    llm = LLMClient({"provider": "mock"})
    prompts = PromptManager("configs/prompts_cot.yaml")
    schema = load_schema("tests/fixtures/schema_example.json")

    agent = CandidateAgent(llm, prompts)
    cset, _, _ = agent.run("a b c", schema)
    assert all(span.start < span.end for span in cset.spans.values())


def test_candidate_agent_span_id_unique_when_llm_reuses_id() -> None:
    llm = LLMClient({"provider": "mock"})
    prompts = PromptManager("configs/prompts_cot.yaml")
    agent = CandidateAgent(llm, prompts)

    payload = {
        "spans": [
            {"span_id": "sp_0001", "start": 0, "end": 3, "text": "Abl"},
            {"span_id": "sp_0001", "start": 14, "end": 18, "text": "trio"},
        ]
    }
    cset = agent._parse_candidate_payload(payload, "Abl regulates trio.")

    assert len(cset.spans) == 2
    assert "sp_0001" in cset.spans
    assert "sp_0002" in cset.spans


def test_candidate_generic_fallback_captures_identifier_like_spans() -> None:
    llm = LLMClient({"provider": "mock"})
    prompts = PromptManager("configs/prompts_cot.yaml")
    schema = load_schema("tests/fixtures/schema_example.json")
    agent = CandidateAgent(llm, prompts)

    text = "System uses module ABC and id X12."
    offsets = agent._generic_fallback_offsets(text, schema)
    tokens = {text[s:e] for s, e in offsets}

    assert "ABC" in tokens
    assert "X12" in tokens
