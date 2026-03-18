from maner.agents.verifier import Verifier
from maner.core.prompting import PromptManager
from maner.core.schema import load_schema
from maner.core.types import Evidence, Mention, Span
from maner.llm.client import LLMClient


def test_verifier_rule_based_drop_invalid() -> None:
    schema = load_schema("tests/fixtures/schema_example.json")
    verifier = Verifier(
        LLMClient({"provider": "mock"}), PromptManager("configs/prompts_cot.yaml"), use_llm=False
    )
    text = "John works at Acme Corp"

    valid = Mention(
        span_id="sp1",
        span=Span(text="John", start=0, end=4),
        ent_type="PERSON",
        confidence=0.8,
        evidence=[Evidence(quote="John", start=0, end=4)],
        rationale="",
    )
    invalid = Mention(
        span_id="sp2",
        span=Span(text="Acme", start=14, end=18),
        ent_type="BAD_TYPE",
        confidence=0.9,
        evidence=[Evidence(quote="Wrong", start=14, end=19)],
        rationale="",
    )

    verified, report, _ = verifier.verify_mentions(text, [valid, invalid], schema)
    assert len(verified) == 1
    assert verified[0].span_id == "sp1"
    assert report["num_input"] == 2
    assert report["num_output"] == 1
