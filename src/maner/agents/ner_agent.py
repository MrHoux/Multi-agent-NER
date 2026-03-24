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
    Mention,
    NERHypothesis,
    Relation,
    Span,
    UsageCost,
    apply_boundary_op,
    is_strict_substring,
)
from maner.llm.client import LLMClient


class NERAgent:
    def __init__(self, llm_client: LLMClient, prompt_manager: PromptManager):
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager

    def run_direct(
        self,
        text: str,
        schema: SchemaDefinition,
    ) -> tuple[NERHypothesis, UsageCost, dict[str, Any]]:
        system, user = self.prompt_manager.render(
            "ner_direct_agent",
            text=text,
            entity_types=schema.to_prompt_block(),
        )

        context = None
        if self.llm_client.provider == "mock":
            context = {
                "mock_result": self._heuristic_direct(
                    text=text,
                    schema=schema,
                )
            }

        llm_result = self.llm_client.chat_json(
            system_prompt=system,
            user_prompt=user,
            task="ner_direct_agent",
            context=context,
        )
        payload = llm_result.parsed_json
        if payload.get("mock") and "result" in payload:
            payload = payload["result"]

        hyp = self._parse_direct_mentions(
            payload=payload,
            text=text,
            valid_types=set(schema.entity_type_names),
        )
        cost = UsageCost(
            calls=llm_result.usage.calls,
            prompt_tokens=llm_result.usage.prompt_tokens,
            completion_tokens=llm_result.usage.completion_tokens,
            total_tokens=llm_result.usage.total_tokens,
            latency_ms=llm_result.usage.latency_ms or [],
        )
        trace = {"agent": "ner_direct", "raw": llm_result.content, "parsed": payload}
        return hyp, cost, trace

    def run_with_expert(
        self,
        text: str,
        candidate_set: CandidateSet,
        schema: SchemaDefinition,
        constraints: ExpertConstraints,
    ) -> tuple[NERHypothesis, UsageCost, dict[str, Any]]:
        system, user = self.prompt_manager.render(
            "ner_expert_agent",
            text=text,
            candidate_spans=self._candidate_payload(candidate_set),
            expert_constraints=self._constraints_payload(constraints),
            entity_types=schema.to_prompt_block(),
        )

        context = None
        if self.llm_client.provider == "mock":
            context = {
                "mock_result": self._heuristic_from_expert(
                    text=text,
                    candidate_set=candidate_set,
                    schema=schema,
                    constraints=constraints,
                )
            }

        llm_result = self.llm_client.chat_json(
            system_prompt=system,
            user_prompt=user,
            task="ner_expert_agent",
            context=context,
        )
        payload = llm_result.parsed_json
        if payload.get("mock") and "result" in payload:
            payload = payload["result"]

        hyp = self._parse_mentions(
            payload=payload,
            text=text,
            candidate_set=candidate_set,
            valid_types=set(schema.entity_type_names),
            source="expert",
        )
        cost = UsageCost(
            calls=llm_result.usage.calls,
            prompt_tokens=llm_result.usage.prompt_tokens,
            completion_tokens=llm_result.usage.completion_tokens,
            total_tokens=llm_result.usage.total_tokens,
            latency_ms=llm_result.usage.latency_ms or [],
        )
        trace = {"agent": "ner_expert", "raw": llm_result.content, "parsed": payload}
        return hyp, cost, trace

    def run_with_context(
        self,
        text: str,
        candidate_set: CandidateSet,
        schema: SchemaDefinition,
        constraints: ExpertConstraints,
    ) -> tuple[NERHypothesis, UsageCost, dict[str, Any]]:
        system, user = self.prompt_manager.render(
            "ner_in_context_agent",
            text=text,
            candidate_spans=self._candidate_payload(candidate_set),
            in_context_constraints=self._constraints_payload(constraints),
            entity_types=schema.to_prompt_block(),
        )

        context = None
        if self.llm_client.provider == "mock":
            context = {
                "mock_result": self._heuristic_from_expert(
                    text=text,
                    candidate_set=candidate_set,
                    schema=schema,
                    constraints=constraints,
                )
            }

        llm_result = self.llm_client.chat_json(
            system_prompt=system,
            user_prompt=user,
            task="ner_in_context_agent",
            context=context,
        )
        payload = llm_result.parsed_json
        if payload.get("mock") and "result" in payload:
            payload = payload["result"]

        hyp = self._parse_mentions(
            payload=payload,
            text=text,
            candidate_set=candidate_set,
            valid_types=set(schema.entity_type_names),
            source="in_context",
        )
        cost = UsageCost(
            calls=llm_result.usage.calls,
            prompt_tokens=llm_result.usage.prompt_tokens,
            completion_tokens=llm_result.usage.completion_tokens,
            total_tokens=llm_result.usage.total_tokens,
            latency_ms=llm_result.usage.latency_ms or [],
        )
        trace = {"agent": "ner_in_context", "raw": llm_result.content, "parsed": payload}
        return hyp, cost, trace

    def run_with_re(
        self,
        text: str,
        candidate_set: CandidateSet,
        schema: SchemaDefinition,
        relations: list[Relation],
    ) -> tuple[NERHypothesis, UsageCost, dict[str, Any]]:
        system, user = self.prompt_manager.render(
            "ner_re_agent",
            text=text,
            candidate_spans=self._candidate_payload(candidate_set),
            relations=self._relations_payload(relations),
            entity_types=schema.to_prompt_block(),
        )

        context = None
        if self.llm_client.provider == "mock":
            context = {
                "mock_result": self._heuristic_from_relations(
                    text=text,
                    candidate_set=candidate_set,
                    schema=schema,
                    relations=relations,
                )
            }

        llm_result = self.llm_client.chat_json(
            system_prompt=system,
            user_prompt=user,
            task="ner_re_agent",
            context=context,
        )
        payload = llm_result.parsed_json
        if payload.get("mock") and "result" in payload:
            payload = payload["result"]

        hyp = self._parse_mentions(
            payload=payload,
            text=text,
            candidate_set=candidate_set,
            valid_types=set(schema.entity_type_names),
            source="re",
        )
        cost = UsageCost(
            calls=llm_result.usage.calls,
            prompt_tokens=llm_result.usage.prompt_tokens,
            completion_tokens=llm_result.usage.completion_tokens,
            total_tokens=llm_result.usage.total_tokens,
            latency_ms=llm_result.usage.latency_ms or [],
        )
        trace = {"agent": "ner_re", "raw": llm_result.content, "parsed": payload}
        return hyp, cost, trace

    def _candidate_payload(self, candidate_set: CandidateSet) -> list[dict[str, Any]]:
        return [
            {"span_id": sid, "text": span.text, "start": span.start, "end": span.end}
            for sid, span in candidate_set.spans.items()
        ]

    def _constraints_payload(self, constraints: ExpertConstraints) -> dict[str, Any]:
        data: dict[str, Any] = {
            "terminology": constraints.terminology,
            "triggers": constraints.triggers,
            "per_span": {},
        }
        for sid, c in constraints.per_span.items():
            data["per_span"][sid] = {
                "candidate_types": c.candidate_types,
                "excluded_types": c.excluded_types,
                "boundary_ops": [{"op": op.op, "params": op.params} for op in c.boundary_ops],
                "evidence": [
                    {"quote": e.quote, "start": e.start, "end": e.end} for e in c.evidence
                ],
                "confidence": c.confidence,
                "rationale": c.rationale,
            }
        return data

    def _relations_payload(self, relations: list[Relation]) -> list[dict[str, Any]]:
        return [
            {
                "head_span_id": r.head_span_id,
                "rel_type": r.rel_type,
                "tail_span_id": r.tail_span_id,
                "confidence": r.confidence,
                "evidence": [
                    {"quote": e.quote, "start": e.start, "end": e.end} for e in r.evidence
                ],
            }
            for r in relations
        ]

    def _heuristic_from_expert(
        self,
        text: str,
        candidate_set: CandidateSet,
        schema: SchemaDefinition,
        constraints: ExpertConstraints,
    ) -> dict[str, Any]:
        mentions = []
        valid = set(schema.entity_type_names)
        for span_id, constraint in constraints.per_span.items():
            if span_id not in candidate_set.spans:
                continue
            span = candidate_set.spans[span_id]
            cand = [
                t
                for t in constraint.candidate_types
                if t in valid and t not in set(constraint.excluded_types)
            ]
            if not cand:
                continue
            ent_type = cand[0]
            evidence = (
                [{"quote": e.quote, "start": e.start, "end": e.end} for e in constraint.evidence]
                if constraint.evidence
                else [{"quote": text[span.start : span.end], "start": span.start, "end": span.end}]
            )
            mentions.append(
                {
                    "span_id": span_id,
                    "ent_type": ent_type,
                    "confidence": max(0.3, float(constraint.confidence)),
                    "evidence": evidence,
                    "rationale": "expert_constraints_alignment",
                    "boundary_ops": [],
                }
            )
        return {"mentions": mentions}

    def _heuristic_direct(
        self,
        text: str,
        schema: SchemaDefinition,
    ) -> dict[str, Any]:
        valid = set(schema.entity_type_names)
        if "PER" in valid:
            from maner.agents.candidate_agent import CandidateAgent

            # Reuse mock candidate spans as a rough direct seed in tests.
            cset = CandidateAgent(self.llm_client, self.prompt_manager).run(text, schema)[0]
            mentions = []
            for span_id, span in cset.spans.items():
                if not span.text[:1].isupper():
                    continue
                mentions.append(
                    {
                        "text": span.text,
                        "start": span.start,
                        "end": span.end,
                        "ent_type": next(iter(valid)),
                        "confidence": 0.6,
                    }
                )
            return {"mentions": mentions}
        return {"mentions": []}

    def _heuristic_from_relations(
        self,
        text: str,
        candidate_set: CandidateSet,
        schema: SchemaDefinition,
        relations: list[Relation],
    ) -> dict[str, Any]:
        label_map: dict[str, str] = {}
        valid = set(schema.entity_type_names)
        for r in relations:
            if r.rel_type == "works_for":
                if "PERSON" in valid:
                    label_map[r.head_span_id] = "PERSON"
                if "ORG" in valid:
                    label_map[r.tail_span_id] = "ORG"
            elif r.rel_type == "located_in":
                if "ORG" in valid:
                    label_map[r.head_span_id] = "ORG"
                if "LOC" in valid:
                    label_map[r.tail_span_id] = "LOC"

        mentions = []
        for span_id, ent_type in label_map.items():
            if span_id not in candidate_set.spans:
                continue
            span = candidate_set.spans[span_id]
            mentions.append(
                {
                    "span_id": span_id,
                    "ent_type": ent_type,
                    "confidence": 0.65,
                    "evidence": [
                        {"quote": text[span.start : span.end], "start": span.start, "end": span.end}
                    ],
                    "rationale": "relation_implied_type",
                    "boundary_ops": [],
                }
            )
        return {"mentions": mentions}

    def _parse_mentions(
        self,
        payload: dict[str, Any],
        text: str,
        candidate_set: CandidateSet,
        valid_types: set[str],
        source: str,
    ) -> NERHypothesis:
        mentions: list[Mention] = []
        for item in payload.get("mentions", []) or []:
            span_id = str(item.get("span_id", ""))
            ent_type = str(item.get("ent_type", ""))
            if span_id not in candidate_set.spans:
                continue
            if ent_type not in valid_types:
                continue

            base_span = candidate_set.spans[span_id]
            spans_to_emit: list[tuple[str, Span]] = [(span_id, base_span)]

            boundary_ops = item.get("boundary_ops", []) or []
            if boundary_ops:
                spans_to_emit = []
                for op_idx, raw_op in enumerate(boundary_ops):
                    op = BoundaryOp(
                        op=str(raw_op.get("op", "")), params=dict(raw_op.get("params", {}))
                    )
                    generated = apply_boundary_op(
                        base_span_id=span_id, base_span=base_span, op=op, text=text
                    )
                    for part_idx, span in enumerate(generated):
                        new_span_id = f"{span_id}__op{op_idx}__{part_idx}"
                        spans_to_emit.append((new_span_id, span))
                if not spans_to_emit:
                    spans_to_emit = [(span_id, base_span)]

            evidence: list[Evidence] = []
            for ev in item.get("evidence", []) or []:
                quote = str(ev.get("quote", ""))
                start = int(ev.get("start", -1))
                end = int(ev.get("end", -1))
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
                        Evidence(
                            quote=text[aligned_start:aligned_end],
                            start=aligned_start,
                            end=aligned_end,
                        )
                    )

            confidence = float(item.get("confidence", 0.0))
            if not evidence:
                confidence = min(confidence, 0.2)
            for sid, span in spans_to_emit:
                mention_evidence = evidence or [
                    Evidence(quote=text[span.start : span.end], start=span.start, end=span.end)
                ]
                mentions.append(
                    Mention(
                        span_id=sid,
                        span=span,
                        ent_type=ent_type,
                        confidence=confidence,
                        evidence=mention_evidence,
                        rationale=f"{source}:{item.get('rationale', '')}".strip(":"),
                    )
                )

        return NERHypothesis(mentions=mentions, source=source)  # type: ignore[arg-type]

    def _parse_direct_mentions(
        self,
        payload: dict[str, Any],
        text: str,
        valid_types: set[str],
    ) -> NERHypothesis:
        mentions: list[Mention] = []
        seen: set[tuple[int, int, str]] = set()
        for idx, item in enumerate(payload.get("mentions", []) or [], start=1):
            if not isinstance(item, dict):
                continue
            ent_type = str(item.get("ent_type", "")).strip()
            if ent_type not in valid_types:
                continue
            quote = str(item.get("text", ""))
            start = int(item.get("start", -1))
            end = int(item.get("end", -1))
            if not is_strict_substring(text, quote, start, end):
                aligned = align_substring_offsets(
                    text=text,
                    quote=quote,
                    start_hint=start,
                    end_hint=end,
                )
                if aligned is None:
                    continue
                start, end = aligned
                quote = text[start:end]
            if start >= end:
                continue
            key = (start, end, ent_type)
            if key in seen:
                continue
            seen.add(key)
            span_id = f"seed_{idx:04d}"
            confidence = float(item.get("confidence", 0.0))
            evidence = [Evidence(quote=quote, start=start, end=end)]
            mentions.append(
                Mention(
                    span_id=span_id,
                    span=Span(
                        text=quote,
                        start=start,
                        end=end,
                        provenance={"op": "DIRECT_SEED"},
                    ),
                    ent_type=ent_type,
                    confidence=confidence,
                    evidence=evidence,
                    rationale=f"direct_seed:{item.get('rationale', '')}".strip(":"),
                )
            )
        return NERHypothesis(mentions=mentions, source="direct")
