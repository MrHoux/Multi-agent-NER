from __future__ import annotations

from typing import Any

from maner.core.prompting import PromptManager
from maner.core.schema import SchemaDefinition
from maner.core.types import Evidence, Mention, Span, UsageCost, is_strict_substring, is_valid_offsets
from maner.llm.client import LLMClient


class DisambiguationAgent:
    def __init__(self, llm_client: LLMClient, prompt_manager: PromptManager):
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager

    def run(
        self,
        text: str,
        mentions: list[Mention],
        schema: SchemaDefinition,
        allow_drop: bool = False,
    ) -> tuple[list[Mention], dict[str, Any], UsageCost]:
        if not mentions:
            return [], {"enabled": True, "input_count": 0, "output_count": 0, "dropped": 0}, UsageCost()

        system, user = self.prompt_manager.render(
            "disambiguation_agent",
            text=text,
            mentions=[
                {
                    "span_id": m.span_id,
                    "text": m.span.text,
                    "start": m.span.start,
                    "end": m.span.end,
                    "ent_type": m.ent_type,
                    "confidence": m.confidence,
                    "rationale": m.rationale,
                    "evidence": [
                        {"quote": ev.quote, "start": ev.start, "end": ev.end}
                        for ev in m.evidence
                    ],
                }
                for m in mentions
            ],
            entity_types=schema.to_prompt_block(),
        )

        context = None
        if self.llm_client.provider == "mock":
            context = {
                "mock_result": {
                    "mentions": [
                        {
                            "source_span_id": m.span_id,
                            "action": "keep",
                            "start": m.span.start,
                            "end": m.span.end,
                            "ent_type": m.ent_type,
                            "confidence": m.confidence,
                            "evidence": [
                                {
                                    "quote": m.span.text,
                                    "start": m.span.start,
                                    "end": m.span.end,
                                }
                            ],
                            "rationale": "mock_keep",
                        }
                        for m in mentions
                    ]
                }
            }

        llm_result = self.llm_client.chat_json(
            system_prompt=system,
            user_prompt=user,
            task="disambiguation_agent",
            context=context,
        )
        payload = llm_result.parsed_json
        if payload.get("mock") and "result" in payload:
            payload = payload["result"]

        by_id = {m.span_id: m for m in mentions}
        out: list[Mention] = []
        dropped = 0
        adjusted = 0
        rejected_adjustments = 0
        seen_source: set[str] = set()

        for item in payload.get("mentions", []) or []:
            source_id = str(item.get("source_span_id", "")).strip()
            if not source_id or source_id not in by_id or source_id in seen_source:
                continue
            seen_source.add(source_id)
            base = by_id[source_id]

            action = str(item.get("action", "keep")).strip().lower()
            if action == "drop":
                if allow_drop:
                    dropped += 1
                    continue
                # Conservative default: keep original mention to avoid recall collapse.
                out.append(base)
                continue

            ent_type = str(item.get("ent_type", base.ent_type))
            start = int(item.get("start", base.span.start))
            end = int(item.get("end", base.span.end))
            if not is_valid_offsets(text, start, end) or start >= end:
                start, end = base.span.start, base.span.end
            quote = text[start:end]

            confidence = float(item.get("confidence", base.confidence))
            confidence = max(0.0, min(1.0, confidence))

            evidence_list: list[Evidence] = []
            for ev in item.get("evidence", []) or []:
                try:
                    ev_quote = str(ev.get("quote", ""))
                    ev_start = int(ev.get("start"))
                    ev_end = int(ev.get("end"))
                except Exception:
                    continue
                if is_valid_offsets(text, ev_start, ev_end) and is_strict_substring(
                    text, ev_quote, ev_start, ev_end
                ):
                    evidence_list.append(Evidence(quote=ev_quote, start=ev_start, end=ev_end))
            if not evidence_list:
                evidence_list = [Evidence(quote=quote, start=start, end=end)]

            rationale = str(item.get("rationale", "")).strip()
            if not rationale:
                rationale = f"disambiguation:{action}"

            if action == "adjust":
                if not _accept_disambiguation_adjustment(
                    text=text,
                    base=base,
                    new_start=start,
                    new_end=end,
                    new_confidence=confidence,
                    evidence=evidence_list,
                ):
                    start = base.span.start
                    end = base.span.end
                    quote = base.span.text
                    confidence = base.confidence
                    evidence_list = list(base.evidence)
                    rationale = f"{rationale}|adjust_rejected_keep_base"
                    rejected_adjustments += 1

            new_span_id = (
                base.span_id
                if (start, end, ent_type) == (base.span.start, base.span.end, base.ent_type)
                else f"{base.span_id}__disamb"
            )
            if new_span_id != base.span_id:
                adjusted += 1

            out.append(
                Mention(
                    span_id=new_span_id,
                    span=Span(
                        text=quote,
                        start=start,
                        end=end,
                        provenance={
                            "op": "DISAMBIGUATION_AGENT",
                            "source_span_id": base.span_id,
                            "action": action,
                        },
                    ),
                    ent_type=ent_type,
                    confidence=confidence,
                    evidence=evidence_list,
                    rationale=f"{base.rationale}|disambiguation:{rationale}",
                )
            )

        # Keep untouched mentions when the agent did not return a decision for them.
        for m in mentions:
            if m.span_id not in seen_source:
                out.append(m)

        usage = UsageCost(
            calls=llm_result.usage.calls,
            prompt_tokens=llm_result.usage.prompt_tokens,
            completion_tokens=llm_result.usage.completion_tokens,
            total_tokens=llm_result.usage.total_tokens,
            latency_ms=llm_result.usage.latency_ms or [],
        )
        trace = {
            "enabled": True,
            "input_count": len(mentions),
            "output_count": len(out),
            "dropped": dropped,
            "adjusted": adjusted,
            "rejected_adjustments": rejected_adjustments,
        }
        return out, trace, usage


def _accept_disambiguation_adjustment(
    text: str,
    base: Mention,
    new_start: int,
    new_end: int,
    new_confidence: float,
    evidence: list[Evidence],
) -> bool:
    if not is_valid_offsets(text, new_start, new_end) or new_start >= new_end:
        return False
    if new_start == base.span.start and new_end == base.span.end:
        return False

    # Boundary rewrites should remain local to the original mention unless very strongly supported.
    overlap = max(new_start, base.span.start) < min(new_end, base.span.end)
    if not overlap:
        return False

    base_len = max(1, base.span.end - base.span.start)
    new_len = max(1, new_end - new_start)
    ratio = float(new_len) / float(base_len)
    conf_delta = float(new_confidence) - float(base.confidence)
    supports_new = any(ev.start == new_start and ev.end == new_end for ev in evidence)

    # Reject very aggressive shrink/expand unless evidence and confidence both clearly support it.
    if ratio < 0.5 or ratio > 1.8:
        if not supports_new or conf_delta < 0.10:
            return False

    # If left-expanding into common discourse wrappers, require stronger signal.
    if new_start < base.span.start:
        left = text[new_start:base.span.start].strip().lower()
        if left and left.isalpha() and len(left) <= 12 and conf_delta < 0.08:
            return False

    # For moderate edits, require either direct evidence support or a meaningful confidence gain.
    if (not supports_new) and conf_delta < 0.05:
        return False
    return True
