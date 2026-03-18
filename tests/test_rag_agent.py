from maner.agents.candidate_agent import CandidateAgent
from maner.agents.rag_agent import RAGAgent
from maner.core.prompting import PromptManager
from maner.core.schema import load_schema
from maner.core.types import is_strict_substring
from maner.llm.client import LLMClient


def test_rag_agent_handoff_structure_and_evidence() -> None:
    text = "John works at Acme Corp in Seattle."
    llm = LLMClient({"provider": "mock"})
    prompts = PromptManager("configs/prompts_cot.yaml")
    schema = load_schema("tests/fixtures/schema_example.json")

    cset, _, _ = CandidateAgent(llm, prompts).run(text, schema)
    memory_items = [
        {
            "kind": "term",
            "key": "Acme Corp",
            "value": {"ent_type": "ORG"},
            "confidence": 0.9,
            "status": "active",
            "seen_count": 3,
            "score": 1.9,
        }
    ]

    handoff, cost, _ = RAGAgent(llm, prompts).run(
        text=text,
        candidate_set=cset,
        schema=schema,
        memory_items=memory_items,
    )

    assert cost.calls == 2
    assert isinstance(handoff.get("handoff_id", ""), str)
    assert isinstance(handoff.get("retrieved_facts", []), list)

    per_span_hints = handoff.get("per_span_hints", {})
    assert set(per_span_hints.keys()).issubset(set(cset.spans.keys()))

    for hint in per_span_hints.values():
        for ev in hint.get("evidence", []):
            assert is_strict_substring(text, ev["quote"], ev["start"], ev["end"])
        assert isinstance(hint.get("sources", []), list)
        assert len(hint.get("sources", [])) >= 1
