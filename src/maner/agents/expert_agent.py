from __future__ import annotations

from typing import Any

from maner.core.alignment import align_substring_offsets
from maner.core.prompting import PromptManager
from maner.core.schema import SchemaDefinition
from maner.core.types import (
    BoundaryOp,
    CandidateSet,
    Evidence,
    ExpertConstraints,
    SpanConstraint,
    UsageCost,
    is_strict_substring,
    is_valid_offsets,
)
from maner.llm.client import LLMClient


class ExpertAgent:
    def __init__(self, llm_client: LLMClient, prompt_manager: PromptManager):
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager

    def plan_retrieval(
        self,
        text: str,
        candidate_set: CandidateSet,
        schema: SchemaDefinition,
        memory_items: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], UsageCost, dict[str, Any]]:
        memory_items = memory_items or []
        system, user = self.prompt_manager.render(
            "expert_retrieval_agent",
            text=text,
            candidate_spans=self._candidate_prompt_payload(candidate_set),
            memory_items=memory_items,
            entity_types=schema.to_prompt_block(),
        )

        context = None
        if self.llm_client.provider == "mock":
            context = {
                "mock_result": self._heuristic_retrieval_plan(
                    text=text,
                    candidate_set=candidate_set,
                    entity_types=schema.entity_type_names,
                )
            }

        llm_result = self.llm_client.chat_json(
            system_prompt=system,
            user_prompt=user,
            task="expert_retrieval_agent",
            context=context,
        )
        payload = llm_result.parsed_json
        if payload.get("mock") and "result" in payload:
            payload = payload["result"]

        retrieval_plan = self._parse_retrieval_plan(payload, candidate_set)
        cost = UsageCost(
            calls=llm_result.usage.calls,
            prompt_tokens=llm_result.usage.prompt_tokens,
            completion_tokens=llm_result.usage.completion_tokens,
            total_tokens=llm_result.usage.total_tokens,
            latency_ms=llm_result.usage.latency_ms or [],
        )
        trace = {"agent": "expert_retrieval", "raw": llm_result.content, "parsed": payload}
        return retrieval_plan, cost, trace

    def run(
        self,
        text: str,
        candidate_set: CandidateSet,
        schema: SchemaDefinition,
        memory_items: list[dict[str, Any]] | None = None,
        rag_handoff: dict[str, Any] | None = None,
        allow_span_proposals: bool = False,
    ) -> tuple[ExpertConstraints, UsageCost, dict[str, Any]]:
        memory_items = memory_items or []
        rag_handoff = rag_handoff or {}
        augmentation_policy = (
            "enabled: you may propose additional recall spans via `new_spans`."
            if allow_span_proposals
            else "disabled: do not output `new_spans`."
        )
        system, user = self.prompt_manager.render(
            "expert_agent",
            text=text,
            candidate_spans=self._candidate_prompt_payload(candidate_set),
            memory_items=memory_items,
            rag_handoff=rag_handoff,
            entity_types=schema.to_prompt_block(),
            augmentation_policy=augmentation_policy,
        )

        context = None
        if self.llm_client.provider == "mock":
            context = {
                "mock_result": self._heuristic_constraints(
                    text=text,
                    candidate_set=candidate_set,
                    entity_types=schema.entity_type_names,
                    memory_items=memory_items,
                    rag_handoff=rag_handoff,
                )
            }

        llm_result = self.llm_client.chat_json(
            system_prompt=system,
            user_prompt=user,
            task="expert_agent",
            context=context,
        )

        payload = llm_result.parsed_json
        if payload.get("mock") and "result" in payload:
            payload = payload["result"]

        constraints = self._parse_constraints(payload, text, candidate_set)
        valid_types = set(schema.entity_type_names)
        span_proposals = (
            self._parse_span_proposals(
                payload,
                text,
                source="expert",
                valid_types=valid_types,
            )
            if allow_span_proposals
            else []
        )
        cost = UsageCost(
            calls=llm_result.usage.calls,
            prompt_tokens=llm_result.usage.prompt_tokens,
            completion_tokens=llm_result.usage.completion_tokens,
            total_tokens=llm_result.usage.total_tokens,
            latency_ms=llm_result.usage.latency_ms or [],
        )
        trace = {
            "agent": "expert",
            "raw": llm_result.content,
            "parsed": payload,
            "rag_ack": payload.get("rag_ack", {}),
            "span_proposals": span_proposals,
        }
        return constraints, cost, trace

    def _candidate_prompt_payload(self, candidate_set: CandidateSet) -> list[dict[str, Any]]:
        return [
            {
                "span_id": sid,
                "text": span.text,
                "start": span.start,
                "end": span.end,
            }
            for sid, span in candidate_set.spans.items()
        ]

    def _heuristic_constraints(
        self,
        text: str,
        candidate_set: CandidateSet,
        entity_types: list[str],
        memory_items: list[dict[str, Any]],
        rag_handoff: dict[str, Any],
    ) -> dict[str, Any]:
        valid_types = set(entity_types)
        rag_hints = (
            rag_handoff.get("per_span_hints", {})
            if isinstance(rag_handoff.get("per_span_hints", {}), dict)
            else {}
        )
        rag_questions = rag_handoff.get("open_questions", []) or []

        per_span: dict[str, Any] = {}
        used_fact_ids: list[str] = []
        for span_id, span in candidate_set.spans.items():
            candidates = list(entity_types)
            excluded: list[str] = []

            hint = rag_hints.get(span_id, {}) if isinstance(rag_hints.get(span_id, {}), dict) else {}
            hinted_types = [
                t for t in hint.get("candidate_types", []) if isinstance(t, str) and t in valid_types
            ]
            if hinted_types:
                for t in hinted_types:
                    if t not in candidates:
                        candidates.append(t)

            per_span[span_id] = {
                "candidate_types": candidates,
                "excluded_types": excluded,
                "boundary_ops": [],
                "evidence": [
                    {"quote": text[span.start : span.end], "start": span.start, "end": span.end}
                ],
                "confidence": 0.7,
            }
            if hinted_types:
                used_fact_ids.append(f"hint::{span_id}")

        terminology = [
            str(item.get("term", "")) for item in memory_items if item.get("kind") == "term"
        ]
        terminology = [t for t in terminology if t]
        triggers: list[str] = []

        return {
            "terminology": terminology,
            "triggers": triggers,
            "per_span": per_span,
            "rag_ack": {
                "handoff_id": str(rag_handoff.get("handoff_id", "")),
                "used_fact_ids": used_fact_ids,
                "answered_questions": [
                    {
                        "q_id": str(q.get("q_id", "")),
                        "answer": "mapped_to_per_span_constraints",
                    }
                    for q in rag_questions[:10]
                    if isinstance(q, dict)
                ],
                "notes": "expert integrated rag hints conservatively",
            },
        }

    def _heuristic_retrieval_plan(
        self,
        text: str,
        candidate_set: CandidateSet,
        entity_types: list[str],
    ) -> dict[str, Any]:
        requests: list[dict[str, Any]] = []
        for idx, (span_id, span) in enumerate(candidate_set.spans.items(), start=1):
            span_text = span.text.strip()
            if not span_text:
                continue
            if len(span_text) < 3 and " " not in span_text:
                continue
            requests.append(
                {
                    "request_id": f"ret_{idx:03d}",
                    "span_ids": [span_id],
                    "question": "Use external reference knowledge only to clarify entity identity or type when local context is insufficient.",
                    "priority": "high" if span_text[:1].isupper() else "medium",
                    "rationale": "possible_type_ambiguity",
                }
            )
        return {
            "retrieval_requests": requests[:8],
            "global_notes": [
                "Request external evidence only for spans whose identity or type remains ambiguous after reading the text."
            ],
            "rationale": "mock retrieval planning",
        }

    def _parse_constraints(
        self,
        payload: dict[str, Any],
        text: str,
        candidate_set: CandidateSet,
    ) -> ExpertConstraints:
        raw_per_span = payload.get("per_span", {})
        constraints = ExpertConstraints(
            terminology=list(payload.get("terminology", []) or []),
            triggers=list(payload.get("triggers", []) or []),
            per_span={},
        )

        for span_id, raw in raw_per_span.items():
            if span_id not in candidate_set.spans:
                continue
            raw_evidence = raw.get("evidence", []) if isinstance(raw, dict) else []
            evidence: list[Evidence] = []
            for item in raw_evidence:
                quote = str(item.get("quote", ""))
                start = int(item.get("start", -1))
                end = int(item.get("end", -1))
                if is_strict_substring(text, quote, start, end):
                    evidence.append(Evidence(quote=quote, start=start, end=end))
                    continue
                aligned = align_substring_offsets(
                    text=text,
                    quote=quote,
                    start_hint=start,
                    end_hint=end,
                )
                if aligned is not None:
                    aligned_start, aligned_end = aligned
                    evidence.append(
                        Evidence(quote=text[aligned_start:aligned_end], start=aligned_start, end=aligned_end)
                    )

            raw_ops = raw.get("boundary_ops", []) if isinstance(raw, dict) else []
            ops: list[BoundaryOp] = []
            for op in raw_ops:
                if not isinstance(op, dict):
                    continue
                ops.append(
                    BoundaryOp(op=str(op.get("op", "")).upper(), params=dict(op.get("params", {})))
                )

            confidence = float(raw.get("confidence", 0.0)) if isinstance(raw, dict) else 0.0
            if not evidence:
                confidence = min(confidence, 0.2)

            constraints.per_span[span_id] = SpanConstraint(
                candidate_types=list(
                    raw.get("candidate_types", []) if isinstance(raw, dict) else []
                ),
                excluded_types=list(raw.get("excluded_types", []) if isinstance(raw, dict) else []),
                boundary_ops=ops,
                evidence=evidence,
                confidence=confidence,
                rationale=str(raw.get("rationale", "") if isinstance(raw, dict) else ""),
            )

        return constraints

    def _parse_retrieval_plan(
        self,
        payload: dict[str, Any],
        candidate_set: CandidateSet,
    ) -> dict[str, Any]:
        requests: list[dict[str, Any]] = []
        for idx, item in enumerate(payload.get("retrieval_requests", []) or [], start=1):
            if not isinstance(item, dict):
                continue
            span_ids = [
                str(span_id)
                for span_id in item.get("span_ids", []) or []
                if str(span_id) in candidate_set.spans
            ]
            if not span_ids:
                continue
            question = str(item.get("question", "")).strip()
            if not question:
                continue
            requests.append(
                {
                    "request_id": str(item.get("request_id", f"ret_{idx:03d}")),
                    "span_ids": span_ids,
                    "question": question,
                    "priority": str(item.get("priority", "medium")),
                    "rationale": str(item.get("rationale", "")),
                }
            )
        return {
            "retrieval_requests": requests[:10],
            "global_notes": [str(x) for x in payload.get("global_notes", []) or []],
            "rationale": str(payload.get("rationale", "")),
        }

    def _parse_span_proposals(
        self,
        payload: dict[str, Any],
        text: str,
        source: str,
        valid_types: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        proposals: list[dict[str, Any]] = []
        for item in payload.get("new_spans", []) or []:
            if not isinstance(item, dict):
                continue
            start = int(item.get("start", -1))
            end = int(item.get("end", -1))

            span_text = str(item.get("text", ""))
            aligned = align_substring_offsets(
                text=text,
                quote=span_text,
                start_hint=start,
                end_hint=end,
            ) if span_text else None
            if aligned is not None:
                start, end = aligned
            elif is_valid_offsets(text, start, end) and start < end:
                span_text = text[start:end]
            else:
                continue
            if not is_valid_offsets(text, start, end) or start >= end:
                continue
            span_text = text[start:end]

            if valid_types is None:
                normalized_hints = [
                    str(t) for t in item.get("type_hints", []) if isinstance(t, str)
                ]
            else:
                normalized_hints = _normalize_type_hints(item.get("type_hints", []), valid_types)
                if not normalized_hints:
                    continue

            evidence: list[dict[str, Any]] = []
            for ev in item.get("evidence", []) or []:
                if not isinstance(ev, dict):
                    continue
                quote = str(ev.get("quote", ""))
                ev_start = int(ev.get("start", -1))
                ev_end = int(ev.get("end", -1))
                if is_strict_substring(text, quote, ev_start, ev_end):
                    evidence.append({"quote": quote, "start": ev_start, "end": ev_end})
                    continue
                ev_aligned = align_substring_offsets(
                    text=text,
                    quote=quote,
                    start_hint=ev_start,
                    end_hint=ev_end,
                )
                if ev_aligned is not None:
                    aligned_start, aligned_end = ev_aligned
                    evidence.append(
                        {
                            "quote": text[aligned_start:aligned_end],
                            "start": aligned_start,
                            "end": aligned_end,
                        }
                    )

            proposals.append(
                {
                    "source": source,
                    "start": start,
                    "end": end,
                    "text": text[start:end],
                    "confidence": float(item.get("confidence", 0.5)),
                    "rationale": str(item.get("rationale", "")),
                    "evidence": evidence,
                    "type_hints": normalized_hints,
                }
            )
        return proposals


def _normalize_type_hints(raw_hints: Any, valid_types: set[str]) -> list[str]:
    if not isinstance(raw_hints, list):
        return []
    canonical = {t.lower(): t for t in valid_types}
    out: list[str] = []
    for hint in raw_hints:
        if not isinstance(hint, str):
            continue
        mapped = canonical.get(hint.lower())
        if mapped and mapped not in out:
            out.append(mapped)
    return out
