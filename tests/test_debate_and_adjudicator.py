from maner.agents.adjudicator_agent import AdjudicatorAgent
from maner.agents.debate_protocol import DebateProtocol
from maner.core.prompting import PromptManager
from maner.core.types import ConflictCluster, Evidence, Mention, NERHypothesis, Span
from maner.llm.client import LLMClient


def _mention(span_id: str, ent_type: str, conf: float) -> Mention:
    text = "John works at Acme Corp"
    start, end = (0, 4) if span_id == "sp1" else (14, 23)
    return Mention(
        span_id=span_id,
        span=Span(text=text[start:end], start=start, end=end),
        ent_type=ent_type,
        confidence=conf,
        evidence=[Evidence(quote=text[start:end], start=start, end=end)],
        rationale="",
    )


def test_debate_protocol_max_turns_and_early_stop() -> None:
    llm = LLMClient({"provider": "mock"})
    prompts = PromptManager("configs/prompts_cot.yaml")
    debate = DebateProtocol(llm, prompts, max_turns=5, epsilon=0.5)

    cluster = {"cluster_id": "c1", "risk_level": "L3", "conflicts": ["type"], "score": 0.8}
    exp_mentions = [_mention("sp1", "PERSON", 0.7)]
    re_mentions = [_mention("sp1", "ORG", 0.69)]

    result = debate.run(
        text="John works at Acme Corp",
        cluster=cluster,
        candidate_set=type("X", (), {"spans": {"sp1": exp_mentions[0].span}})(),
        exp_mentions=exp_mentions,
        re_mentions=re_mentions,
    )

    assert 1 <= len(result.rounds) <= 5
    assert result.usage.debate_turns == len(result.rounds)


def test_adjudicator_uses_debate_for_l3() -> None:
    llm = LLMClient({"provider": "mock"})
    prompts = PromptManager("configs/prompts_cot.yaml")
    debate = DebateProtocol(llm, prompts, max_turns=5, epsilon=0.01)
    adj = AdjudicatorAgent(llm, prompts)

    y_exp = NERHypothesis(mentions=[_mention("sp1", "PERSON", 0.75)], source="expert")
    y_re = NERHypothesis(mentions=[_mention("sp1", "ORG", 0.74)], source="re")
    clusters = [
        ConflictCluster(
            cluster_id="cluster_0001",
            span_ids=["sp1"],
            conflicts=["type"],
            risk_level="L3",
            score=0.9,
        )
    ]

    decision, usage, trace = adj.run(
        text="John works at Acme Corp",
        clusters=clusters,
        y_exp=y_exp,
        y_re=y_re,
        debate_protocol=debate,
        enable_debate=True,
        l3_only=True,
    )

    assert decision.final_mentions
    assert usage.debate_triggered == 1
    assert trace["clusters"][0]["decision"] == "debate"


def test_adjudicator_conservative_singleton_can_abstain() -> None:
    llm = LLMClient({"provider": "mock"})
    prompts = PromptManager("configs/prompts_cot.yaml")
    debate = DebateProtocol(llm, prompts, max_turns=5, epsilon=0.01)
    adj = AdjudicatorAgent(llm, prompts)

    weak = Mention(
        span_id="sp1",
        span=Span(text="finding", start=0, end=7),
        ent_type="ENTITY",
        confidence=0.85,
        evidence=[Evidence(quote="finding", start=0, end=7)],
        rationale="",
    )
    y_exp = NERHypothesis(mentions=[weak], source="expert")
    y_re = NERHypothesis(mentions=[], source="re")
    clusters = [
        ConflictCluster(
            cluster_id="cluster_0001",
            span_ids=["sp1"],
            conflicts=["existence"],
            risk_level="L1",
            score=0.35,
        )
    ]

    decision, _, trace = adj.run(
        text="finding observed",
        clusters=clusters,
        y_exp=y_exp,
        y_re=y_re,
        debate_protocol=debate,
        enable_debate=False,
        l3_only=True,
        singleton_policy="conservative",
        singleton_min_confidence=0.9,
        singleton_require_entity_like=True,
    )

    assert decision.final_mentions == []
    assert trace["clusters"][0]["decision"] == "abstain_expert_only"
