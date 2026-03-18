from maner.agents.candidate_agent import CandidateAgent
from maner.agents.re_agent import REAgent
from maner.core.prompting import PromptManager
from maner.core.schema import load_schema
from maner.core.types import CandidateSet, Evidence, Relation, Span, is_strict_substring
from maner.llm.client import LLMClient


def test_re_agent_relations_with_evidence() -> None:
    text = "John works at Acme Corp in Seattle."
    llm = LLMClient({"provider": "mock"})
    prompts = PromptManager("configs/prompts_cot.yaml")
    schema = load_schema("tests/fixtures/schema_example.json")

    cset, _, _ = CandidateAgent(llm, prompts).run(text, schema)
    relations, cost, _ = REAgent(llm, prompts).run(text, cset, schema, memory_items=[])

    assert cost.calls == 1
    assert isinstance(relations, list)
    if relations:
        rel = relations[0]
        assert rel.head_span_id in cset.spans
        assert rel.tail_span_id in cset.spans
        for ev in rel.evidence:
            assert is_strict_substring(text, ev.quote, ev.start, ev.end)


def test_re_coordination_policy_prefers_pair_and_targets() -> None:
    text = "trio and Abl cooperate in regulating axon outgrowth."
    agent = REAgent(LLMClient({"provider": "mock"}), PromptManager("configs/prompts_cot.yaml"))

    cset = CandidateSet(
        has_entity=True,
        spans={
            "sp_0001": Span(text="trio", start=0, end=4),
            "sp_0002": Span(text="Abl", start=9, end=12),
            "sp_0003": Span(text="axon outgrowth", start=37, end=51),
        },
    )
    baseline = [
        Relation(
            head_span_id="sp_0001",
            rel_type="regulates",
            tail_span_id="sp_0002",
            confidence=0.6,
            evidence=[Evidence(quote="trio and Abl", start=0, end=12)],
        )
    ]

    out = agent._apply_coordination_policy(
        text=text,
        candidate_set=cset,
        relations=baseline,
        relation_constraints=None,
    )

    keys = {(r.head_span_id, r.rel_type, r.tail_span_id) for r in out}
    assert ("sp_0001", "cooperates_with", "sp_0002") in keys
    assert ("sp_0001", "regulates", "sp_0003") in keys
    assert ("sp_0002", "regulates", "sp_0003") in keys
    assert ("sp_0001", "regulates", "sp_0002") not in keys
