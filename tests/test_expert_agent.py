from maner.agents.candidate_agent import CandidateAgent
from maner.agents.expert_agent import ExpertAgent
from maner.core.prompting import PromptManager
from maner.core.schema import load_schema
from maner.core.types import is_strict_substring
from maner.llm.client import LLMClient


def test_expert_agent_constraints_with_evidence() -> None:
    text = "John works at Acme Corp in Seattle."
    llm = LLMClient({"provider": "mock"})
    prompts = PromptManager("configs/prompts_cot.yaml")
    schema = load_schema("tests/fixtures/schema_example.json")

    c_agent = CandidateAgent(llm, prompts)
    cset, _, _ = c_agent.run(text, schema)

    e_agent = ExpertAgent(llm, prompts)
    constraints, cost, _ = e_agent.run(text, cset, schema, memory_items=[])

    assert cost.calls == 1
    assert constraints.per_span
    assert set(constraints.per_span.keys()).issubset(set(cset.spans.keys()))

    for c in constraints.per_span.values():
        assert isinstance(c.candidate_types, list)
        for ev in c.evidence:
            assert is_strict_substring(text, ev.quote, ev.start, ev.end)


def test_expert_agent_accepts_rag_handoff() -> None:
    text = "John works at Acme Corp in Seattle."
    llm = LLMClient({"provider": "mock"})
    prompts = PromptManager("configs/prompts_cot.yaml")
    schema = load_schema("tests/fixtures/schema_example.json")

    cset, _, _ = CandidateAgent(llm, prompts).run(text, schema)
    first_span_id = next(iter(cset.spans.keys()))
    first_span = cset.spans[first_span_id]
    rag_handoff = {
        "handoff_id": "rag_handoff_test",
        "per_span_hints": {
            first_span_id: {
                "candidate_types": ["ORG"],
                "evidence": [
                    {
                        "quote": text[first_span.start : first_span.end],
                        "start": first_span.start,
                        "end": first_span.end,
                    }
                ],
                "query_note": "test_hint",
            }
        },
        "open_questions": [{"q_id": "q1", "span_id": first_span_id, "question": "type?", "priority": "high"}],
    }

    _, _, trace = ExpertAgent(llm, prompts).run(
        text=text,
        candidate_set=cset,
        schema=schema,
        memory_items=[],
        rag_handoff=rag_handoff,
    )
    assert "rag_ack" in trace
