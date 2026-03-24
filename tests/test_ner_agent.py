from maner.agents.candidate_agent import CandidateAgent
from maner.agents.expert_agent import ExpertAgent
from maner.agents.ner_agent import NERAgent
from maner.agents.re_agent import REAgent
from maner.core.prompting import PromptManager
from maner.core.schema import load_schema
from maner.core.types import is_strict_substring
from maner.llm.client import LLMClient


def test_ner_agent_routes_output_valid_mentions() -> None:
    text = "John works at Acme Corp in Seattle."
    schema = load_schema("tests/fixtures/schema_example.json")
    llm = LLMClient({"provider": "mock"})
    prompts = PromptManager("configs/prompts_cot.yaml")

    cset, _, _ = CandidateAgent(llm, prompts).run(text, schema)
    constraints, _, _ = ExpertAgent(llm, prompts).run(text, cset, schema, [])
    relations, _, _, _ = REAgent(llm, prompts).run(text, cset, schema, [])

    ner_agent = NERAgent(llm, prompts)
    y_exp, _, _ = ner_agent.run_with_expert(text, cset, schema, constraints)
    y_re, _, _ = ner_agent.run_with_re(text, cset, schema, relations)

    valid_types = set(schema.entity_type_names)
    assert y_exp.source == "expert"
    assert y_re.source == "re"

    for hyp in [y_exp, y_re]:
        for m in hyp.mentions:
            assert m.ent_type in valid_types
            assert m.evidence
            for ev in m.evidence:
                assert is_strict_substring(text, ev.quote, ev.start, ev.end)
