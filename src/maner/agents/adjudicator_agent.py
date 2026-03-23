from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from maner.agents.debate_protocol import DebateProtocol
from maner.core.prompting import PromptManager
from maner.core.types import ConflictCluster, Decision, Mention, NERHypothesis, UsageCost
from maner.llm.client import LLMClient


class AdjudicatorAgent:
    def __init__(self, llm_client: LLMClient, prompt_manager: PromptManager):
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager

    def run(
        self,
        text: str,
        clusters: list[ConflictCluster],
        y_exp: NERHypothesis,
        y_re: NERHypothesis,
        debate_protocol: DebateProtocol,
        enable_debate: bool = True,
        l3_only: bool = True,
        review_all_mentions: bool = True,
        singleton_policy: str = "legacy",
        singleton_min_confidence: float = 0.0,
        singleton_require_entity_like: bool = False,
    ) -> tuple[Decision, UsageCost, dict[str, Any]]:
        exp_by_sid = {m.span_id: m for m in y_exp.mentions}
        re_by_sid = {m.span_id: m for m in y_re.mentions}

        selected_mentions: dict[str, Mention] = {}
        trace: list[dict[str, Any]] = []
        usage = UsageCost()

        for cluster in clusters:
            exp_mentions = [exp_by_sid[sid] for sid in cluster.span_ids if sid in exp_by_sid]
            re_mentions = [re_by_sid[sid] for sid in cluster.span_ids if sid in re_by_sid]

            debate_required = enable_debate and ((not l3_only) or cluster.risk_level == "L3")
            winner, cluster_trace, cluster_usage = self._decide_cluster(
                text=text,
                cluster=cluster,
                exp_mentions=exp_mentions,
                re_mentions=re_mentions,
                debate_protocol=debate_protocol,
                debate_required=debate_required,
                review_all_mentions=review_all_mentions,
                singleton_policy=singleton_policy,
                singleton_min_confidence=singleton_min_confidence,
                singleton_require_entity_like=singleton_require_entity_like,
            )

            usage.calls += cluster_usage.calls
            usage.prompt_tokens = _sum_opt(usage.prompt_tokens, cluster_usage.prompt_tokens)
            usage.completion_tokens = _sum_opt(
                usage.completion_tokens, cluster_usage.completion_tokens
            )
            usage.total_tokens = _sum_opt(usage.total_tokens, cluster_usage.total_tokens)
            usage.latency_ms.extend(cluster_usage.latency_ms)
            usage.debate_turns += cluster_usage.debate_turns
            usage.debate_triggered += cluster_usage.debate_triggered

            if winner is not None:
                selected_mentions[winner.span_id] = winner
            trace.append(cluster_trace)

        final_mentions = list(selected_mentions.values())
        return Decision(final_mentions=final_mentions, trace=trace), usage, {"clusters": trace}

    def _decide_cluster(
        self,
        text: str,
        cluster: ConflictCluster,
        exp_mentions: list[Mention],
        re_mentions: list[Mention],
        debate_protocol: DebateProtocol,
        debate_required: bool,
        review_all_mentions: bool,
        singleton_policy: str,
        singleton_min_confidence: float,
        singleton_require_entity_like: bool,
    ) -> tuple[Mention | None, dict[str, Any], UsageCost]:
        usage = UsageCost()

        if not exp_mentions and not re_mentions:
            return None, {"cluster_id": cluster.cluster_id, "decision": "none"}, usage

        exp_best = max(exp_mentions, key=lambda m: m.confidence) if exp_mentions else None
        re_best = max(re_mentions, key=lambda m: m.confidence) if re_mentions else None

        if exp_best and not re_best:
            if review_all_mentions:
                return self._semantic_review(
                    text=text,
                    cluster=cluster,
                    exp_mentions=exp_mentions,
                    re_mentions=[],
                )
            if singleton_policy == "conservative" and not _singleton_passes(
                text=text,
                mention=exp_best,
                min_confidence=singleton_min_confidence,
                require_entity_like=singleton_require_entity_like,
            ):
                return (
                    None,
                    {
                        "cluster_id": cluster.cluster_id,
                        "decision": "abstain_expert_only",
                        "reason": "singleton_guard_failed",
                    },
                    usage,
                )
            winner = replace(exp_best, rationale=f"adjudicator:expert_only:{cluster.cluster_id}")
            return winner, {"cluster_id": cluster.cluster_id, "decision": "expert_only"}, usage
        if re_best and not exp_best:
            if review_all_mentions:
                return self._semantic_review(
                    text=text,
                    cluster=cluster,
                    exp_mentions=[],
                    re_mentions=re_mentions,
                )
            if singleton_policy == "conservative" and not _singleton_passes(
                text=text,
                mention=re_best,
                min_confidence=singleton_min_confidence,
                require_entity_like=singleton_require_entity_like,
            ):
                return (
                    None,
                    {
                        "cluster_id": cluster.cluster_id,
                        "decision": "abstain_re_only",
                        "reason": "singleton_guard_failed",
                    },
                    usage,
                )
            winner = replace(re_best, rationale=f"adjudicator:re_only:{cluster.cluster_id}")
            return winner, {"cluster_id": cluster.cluster_id, "decision": "re_only"}, usage

        assert exp_best is not None and re_best is not None

        if debate_required:
            debate_result = debate_protocol.run(
                text=text,
                cluster={
                    "cluster_id": cluster.cluster_id,
                    "conflicts": cluster.conflicts,
                    "risk_level": cluster.risk_level,
                    "score": cluster.score,
                },
                candidate_set=_mentions_to_candidate(exp_mentions + re_mentions),
                exp_mentions=exp_mentions,
                re_mentions=re_mentions,
            )
            usage = debate_result.usage
            if debate_result.winner_source == "expert":
                winner = replace(
                    exp_best,
                    ent_type=debate_result.winner_ent_type or exp_best.ent_type,
                    confidence=max(exp_best.confidence, debate_result.winner_confidence),
                    rationale=f"adjudicator:debate:expert:{cluster.cluster_id}:t{len(debate_result.rounds)}",
                )
            else:
                winner = replace(
                    re_best,
                    ent_type=debate_result.winner_ent_type or re_best.ent_type,
                    confidence=max(re_best.confidence, debate_result.winner_confidence),
                    rationale=f"adjudicator:debate:re:{cluster.cluster_id}:t{len(debate_result.rounds)}",
                )
            trace = {
                "cluster_id": cluster.cluster_id,
                "decision": "debate",
                "winner": winner.span_id,
                "winner_source": debate_result.winner_source,
                "rounds": debate_result.rounds,
            }
            return winner, trace, usage

        if review_all_mentions:
            return self._semantic_review(
                text=text,
                cluster=cluster,
                exp_mentions=exp_mentions,
                re_mentions=re_mentions,
            )

        winner = exp_best if exp_best.confidence >= re_best.confidence else re_best
        source = "expert" if winner is exp_best else "re"
        winner = replace(winner, rationale=f"adjudicator:confidence:{source}:{cluster.cluster_id}")
        trace = {
            "cluster_id": cluster.cluster_id,
            "decision": "confidence",
            "winner": winner.span_id,
        }
        return winner, trace, usage

    def _semantic_review(
        self,
        text: str,
        cluster: ConflictCluster,
        exp_mentions: list[Mention],
        re_mentions: list[Mention],
    ) -> tuple[Mention | None, dict[str, Any], UsageCost]:
        system, user = self.prompt_manager.render(
            "adjudicator_agent",
            text=text,
            cluster={
                "cluster_id": cluster.cluster_id,
                "conflicts": cluster.conflicts,
                "risk_level": cluster.risk_level,
                "score": cluster.score,
            },
            expert_mentions=[_mention_payload(m) for m in exp_mentions],
            re_mentions=[_mention_payload(m) for m in re_mentions],
        )

        context = None
        if self.llm_client.provider == "mock":
            context = {
                "mock_result": self._heuristic_semantic_review(
                    cluster=cluster,
                    exp_mentions=exp_mentions,
                    re_mentions=re_mentions,
                )
            }

        llm_result = self.llm_client.chat_json(
            system_prompt=system,
            user_prompt=user,
            task="adjudicator_agent",
            context=context,
        )

        payload = llm_result.parsed_json
        if payload.get("mock") and "result" in payload:
            payload = payload["result"]

        usage = UsageCost(
            calls=llm_result.usage.calls,
            prompt_tokens=llm_result.usage.prompt_tokens,
            completion_tokens=llm_result.usage.completion_tokens,
            total_tokens=llm_result.usage.total_tokens,
            latency_ms=llm_result.usage.latency_ms or [],
        )

        winner_source = str(payload.get("winner_source", "none"))
        winner_span_id = str(payload.get("winner_span_id", ""))
        winner_ent_type = str(payload.get("winner_ent_type", ""))
        winner_confidence = float(payload.get("confidence", 0.0) or 0.0)
        rationale = str(payload.get("rationale", ""))

        exp_by_sid = {m.span_id: m for m in exp_mentions}
        re_by_sid = {m.span_id: m for m in re_mentions}
        winner: Mention | None = None
        decision = "abstain"

        if winner_source == "expert" and winner_span_id in exp_by_sid:
            base = exp_by_sid[winner_span_id]
            decision = "semantic_keep"
            winner = replace(
                base,
                ent_type=winner_ent_type or base.ent_type,
                confidence=max(0.0, min(1.0, max(base.confidence, winner_confidence))),
                rationale=f"adjudicator:semantic:expert:{cluster.cluster_id}|{rationale}".strip("|"),
            )
        elif winner_source == "re" and winner_span_id in re_by_sid:
            base = re_by_sid[winner_span_id]
            decision = "semantic_keep"
            winner = replace(
                base,
                ent_type=winner_ent_type or base.ent_type,
                confidence=max(0.0, min(1.0, max(base.confidence, winner_confidence))),
                rationale=f"adjudicator:semantic:re:{cluster.cluster_id}|{rationale}".strip("|"),
            )

        trace = {
            "cluster_id": cluster.cluster_id,
            "decision": decision,
            "winner": winner.span_id if winner is not None else None,
            "winner_source": winner_source,
            "review_mode": "semantic",
        }
        return winner, trace, usage

    def _heuristic_semantic_review(
        self,
        cluster: ConflictCluster,
        exp_mentions: list[Mention],
        re_mentions: list[Mention],
    ) -> dict[str, Any]:
        candidates: list[tuple[str, Mention]] = [("expert", m) for m in exp_mentions] + [
            ("re", m) for m in re_mentions
        ]
        if not candidates:
            return {
                "winner_source": "none",
                "winner_span_id": "",
                "winner_ent_type": "",
                "confidence": 0.0,
                "rationale": "no_candidate",
            }

        filtered = [
            (source, mention)
            for source, mention in candidates
            if mention.evidence and mention.confidence >= 0.2
        ]
        if not filtered:
            return {
                "winner_source": "none",
                "winner_span_id": "",
                "winner_ent_type": "",
                "confidence": 0.0,
                "rationale": "no_semantic_support",
            }

        best_source, best_mention = max(filtered, key=lambda item: item[1].confidence)
        return {
            "winner_source": best_source,
            "winner_span_id": best_mention.span_id,
            "winner_ent_type": best_mention.ent_type,
            "confidence": best_mention.confidence,
            "rationale": f"heuristic_semantic_review:{cluster.cluster_id}",
        }


def _mentions_to_candidate(mentions: list[Mention]):
    from maner.core.types import CandidateSet

    return CandidateSet(has_entity=bool(mentions), spans={m.span_id: m.span for m in mentions})


def _mention_payload(mention: Mention) -> dict[str, Any]:
    return {
        "span_id": mention.span_id,
        "text": mention.span.text,
        "start": mention.span.start,
        "end": mention.span.end,
        "ent_type": mention.ent_type,
        "confidence": mention.confidence,
        "evidence": [
            {"quote": ev.quote, "start": ev.start, "end": ev.end} for ev in mention.evidence
        ],
        "rationale": mention.rationale,
    }


def _sum_opt(a: int | None, b: int | None) -> int | None:
    if a is None and b is None:
        return None
    return int(a or 0) + int(b or 0)


def _singleton_passes(
    text: str,
    mention: Mention,
    min_confidence: float,
    require_entity_like: bool,
) -> bool:
    if mention.confidence < min_confidence:
        near_threshold = mention.confidence >= max(0.0, min_confidence - 0.08)
        if not near_threshold:
            return False
        if not _looks_entity_like(text, mention):
            return False
    if not mention.evidence:
        return False
    if require_entity_like:
        if not _looks_entity_like(text, mention):
            return False
    return True


def _looks_entity_like(text: str, mention: Mention) -> bool:
    stripped = mention.span.text.strip()
    if not stripped:
        return False

    tokens = re.findall(r"[A-Za-z0-9]+", stripped)
    if not tokens:
        return False
    if len(tokens) > 14:
        return False

    lower_tokens = [tok.lower() for tok in tokens]
    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "for",
        "with",
        "from",
        "to",
        "of",
        "in",
        "on",
        "by",
        "is",
        "are",
        "was",
        "were",
        "before",
        "during",
        "after",
        "while",
        "when",
        "where",
        "which",
        "that",
        "this",
        "these",
        "those",
        "treatment",
        "patient",
        "patients",
        "study",
        "studies",
        "result",
        "results",
        "approach",
        "methods",
        "method",
        "level",
        "levels",
        "stage",
        "stages",
        "group",
        "groups",
    }
    if all(tok in stop for tok in lower_tokens):
        return False
    if all(tok.isdigit() for tok in tokens):
        return False

    has_digit = any(any(ch.isdigit() for ch in tok) for tok in tokens)
    has_caps = any(any(ch.isupper() for ch in tok) for tok in tokens)
    has_mixed_case_token = any(
        any(ch.isupper() for ch in tok) and any(ch.islower() for ch in tok)
        for tok in tokens
    )
    has_connector = any(ch in stripped for ch in {"-", "_", "/", "'", "+"})
    if has_digit or has_caps or has_mixed_case_token or has_connector:
        return True

    if len(tokens) == 1:
        tok = tokens[0]
        lower = tok.lower()
        # Sentence-initial title-case nouns are high-FP in singleton clusters.
        if tok[:1].isupper() and tok[1:].islower() and mention.span.start == 0:
            return False
        if lower in stop:
            return False
        return False

    meaningful_tokens = [tok for tok in lower_tokens if tok not in stop]
    if len(meaningful_tokens) < 2:
        return False
    alpha_tokens = [tok for tok in tokens if tok.isalpha()]
    if alpha_tokens and all(tok[:1].isupper() and tok[1:].islower() for tok in alpha_tokens):
        return True
    if any(len(tok) >= 4 for tok in meaningful_tokens):
        return True
    return False
