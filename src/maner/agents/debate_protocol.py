from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from maner.core.prompting import PromptManager
from maner.core.types import CandidateSet, Mention, UsageCost
from maner.llm.client import LLMClient


@dataclass
class DebateResult:
    winner_source: str
    winner_confidence: float
    winner_ent_type: str
    rounds: list[dict[str, Any]]
    usage: UsageCost


class DebateProtocol:
    def __init__(
        self,
        llm_client: LLMClient,
        prompt_manager: PromptManager,
        max_turns: int = 5,
        epsilon: float = 0.01,
    ):
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager
        self.max_turns = max_turns
        self.epsilon = epsilon

    def run(
        self,
        text: str,
        cluster: dict[str, Any],
        candidate_set: CandidateSet,
        exp_mentions: list[Mention],
        re_mentions: list[Mention],
    ) -> DebateResult:
        rounds: list[dict[str, Any]] = []
        usage = UsageCost()
        last_label = None
        last_conf = 0.0
        evidence_bank: set[tuple[str, int, int]] = set()

        for turn in range(1, self.max_turns + 1):
            round_payload = self._build_round_payload(
                turn, text, exp_mentions, re_mentions, evidence_bank
            )
            system, user = self.prompt_manager.render(
                "debate_agent",
                text=text,
                cluster=cluster,
                history=rounds,
                candidate_spans=[
                    {"span_id": sid, "text": span.text, "start": span.start, "end": span.end}
                    for sid, span in candidate_set.spans.items()
                ],
            )

            context = {"mock_result": round_payload} if self.llm_client.provider == "mock" else None
            llm_result = self.llm_client.chat_json(
                system_prompt=system,
                user_prompt=user,
                task="debate_agent",
                context=context,
            )
            payload = llm_result.parsed_json
            if payload.get("mock") and "result" in payload:
                payload = payload["result"]

            usage.calls += llm_result.usage.calls
            usage.prompt_tokens = _sum_opt(usage.prompt_tokens, llm_result.usage.prompt_tokens)
            usage.completion_tokens = _sum_opt(
                usage.completion_tokens, llm_result.usage.completion_tokens
            )
            usage.total_tokens = _sum_opt(usage.total_tokens, llm_result.usage.total_tokens)
            usage.latency_ms.extend(llm_result.usage.latency_ms or [])

            evidences = payload.get("evidence", []) or []
            new_evidence_count = 0
            for ev in evidences:
                k = (str(ev.get("quote", "")), int(ev.get("start", -1)), int(ev.get("end", -1)))
                if k not in evidence_bank:
                    evidence_bank.add(k)
                    new_evidence_count += 1

            claim_label = str(payload.get("ent_type", ""))
            claim_conf = float(payload.get("confidence", 0.0))

            rounds.append(
                {
                    "turn": turn,
                    "claim_source": str(payload.get("claim_source", "expert")),
                    "ent_type": claim_label,
                    "confidence": claim_conf,
                    "counter_argument": str(payload.get("counter_argument", "")),
                    "evidence": evidences,
                    "new_evidence_count": new_evidence_count,
                }
            )

            stable = claim_label == last_label and last_label is not None
            low_gain = abs(claim_conf - last_conf) < self.epsilon
            if new_evidence_count == 0 or stable or low_gain:
                break
            last_label = claim_label
            last_conf = claim_conf

        winner = max(rounds, key=lambda x: x["confidence"]) if rounds else None
        if winner is None:
            return DebateResult(
                winner_source="expert",
                winner_confidence=0.0,
                winner_ent_type="",
                rounds=[],
                usage=usage,
            )

        usage.debate_turns = len(rounds)
        usage.debate_triggered = 1
        return DebateResult(
            winner_source=winner["claim_source"],
            winner_confidence=float(winner["confidence"]),
            winner_ent_type=winner["ent_type"],
            rounds=rounds,
            usage=usage,
        )

    def _build_round_payload(
        self,
        turn: int,
        text: str,
        exp_mentions: list[Mention],
        re_mentions: list[Mention],
        evidence_bank: set[tuple[str, int, int]],
    ) -> dict[str, Any]:
        exp_best = max(exp_mentions, key=lambda m: m.confidence) if exp_mentions else None
        re_best = max(re_mentions, key=lambda m: m.confidence) if re_mentions else None

        if turn % 2 == 1:
            pick = (
                exp_best
                if exp_best and (not re_best or exp_best.confidence >= re_best.confidence)
                else re_best
            )
            source = "expert" if pick is exp_best else "re"
        else:
            pick = re_best if re_best else exp_best
            source = "re" if pick is re_best else "expert"

        if pick is None:
            return {
                "claim_source": "expert",
                "ent_type": "",
                "confidence": 0.0,
                "counter_argument": "insufficient_mentions",
                "evidence": [],
            }

        ev = pick.evidence[0] if pick.evidence else None
        ev_payload = [{"quote": ev.quote, "start": ev.start, "end": ev.end}] if ev else []
        return {
            "claim_source": source,
            "ent_type": pick.ent_type,
            "confidence": max(0.0, min(1.0, pick.confidence + 0.01 * turn)),
            "counter_argument": "confidence_and_evidence_based",
            "evidence": ev_payload,
        }


def _sum_opt(a: int | None, b: int | None) -> int | None:
    if a is None and b is None:
        return None
    return int(a or 0) + int(b or 0)
