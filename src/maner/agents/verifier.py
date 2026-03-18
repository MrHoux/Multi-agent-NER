from __future__ import annotations

from dataclasses import replace
from typing import Any

from maner.core.prompting import PromptManager
from maner.core.schema import SchemaDefinition
from maner.core.types import Mention, UsageCost, is_strict_substring, is_valid_offsets
from maner.llm.client import LLMClient


class Verifier:
    def __init__(
        self,
        llm_client: LLMClient,
        prompt_manager: PromptManager,
        use_llm: bool = False,
        strict_drop_invalid: bool = True,
    ):
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager
        self.use_llm = use_llm
        self.strict_drop_invalid = strict_drop_invalid

    def verify_mentions(
        self,
        text: str,
        mentions: list[Mention],
        schema: SchemaDefinition,
    ) -> tuple[list[Mention], dict[str, Any], UsageCost]:
        valid_types = set(schema.entity_type_names)
        verified: list[Mention] = []
        report_items: list[dict[str, Any]] = []
        usage = UsageCost()

        for m in mentions:
            reasons: list[str] = []

            if not is_valid_offsets(text, m.span.start, m.span.end):
                reasons.append("invalid_offsets")
            if m.ent_type not in valid_types:
                reasons.append("invalid_ent_type")
            if not m.evidence:
                reasons.append("missing_evidence")
            else:
                for ev in m.evidence:
                    if not is_strict_substring(text, ev.quote, ev.start, ev.end):
                        reasons.append("invalid_evidence_substring")
                        break

            llm_adjustment = None
            llm_pass = True
            if self.use_llm:
                llm_pass, llm_adjustment, llm_reason, llm_usage = self._llm_verify(text, m, schema)
                usage.calls += llm_usage.calls
                usage.prompt_tokens = _sum_opt(usage.prompt_tokens, llm_usage.prompt_tokens)
                usage.completion_tokens = _sum_opt(
                    usage.completion_tokens, llm_usage.completion_tokens
                )
                usage.total_tokens = _sum_opt(usage.total_tokens, llm_usage.total_tokens)
                usage.latency_ms.extend(llm_usage.latency_ms)
                if not llm_pass:
                    reasons.append(f"llm_fail:{llm_reason}")

            passed = len(reasons) == 0 and llm_pass
            adjusted = m
            if llm_adjustment is not None:
                adjusted = replace(adjusted, confidence=max(0.0, min(1.0, llm_adjustment)))

            if passed or not self.strict_drop_invalid:
                if not passed:
                    adjusted = replace(adjusted, confidence=min(adjusted.confidence, 0.2))
                verified.append(adjusted)

            report_items.append(
                {
                    "span_id": m.span_id,
                    "passed": passed,
                    "reasons": reasons,
                    "confidence_before": m.confidence,
                    "confidence_after": adjusted.confidence,
                }
            )

        pass_rate = (
            (sum(1 for x in report_items if x["passed"]) / len(report_items))
            if report_items
            else 1.0
        )
        report = {
            "num_input": len(mentions),
            "num_output": len(verified),
            "pass_rate": pass_rate,
            "items": report_items,
        }
        return verified, report, usage

    def _llm_verify(
        self,
        text: str,
        mention: Mention,
        schema: SchemaDefinition,
    ) -> tuple[bool, float | None, str, UsageCost]:
        system, user = self.prompt_manager.render(
            "llm_verifier",
            text=text,
            mentions=[
                {
                    "span_id": mention.span_id,
                    "text": mention.span.text,
                    "start": mention.span.start,
                    "end": mention.span.end,
                    "ent_type": mention.ent_type,
                    "confidence": mention.confidence,
                    "evidence": [
                        {"quote": ev.quote, "start": ev.start, "end": ev.end}
                        for ev in mention.evidence
                    ],
                }
            ],
            entity_types=schema.to_prompt_block(),
        )

        context = None
        if self.llm_client.provider == "mock":
            context = {
                "mock_result": {
                    "pass": True,
                    "adjusted_confidence": mention.confidence,
                    "reason": "mock_pass",
                }
            }

        llm_result = self.llm_client.chat_json(
            system_prompt=system,
            user_prompt=user,
            task="llm_verifier",
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
        return (
            bool(payload.get("pass", False)),
            float(payload.get("adjusted_confidence"))
            if payload.get("adjusted_confidence") is not None
            else None,
            str(payload.get("reason", "")),
            usage,
        )


def _sum_opt(a: int | None, b: int | None) -> int | None:
    if a is None and b is None:
        return None
    return int(a or 0) + int(b or 0)
