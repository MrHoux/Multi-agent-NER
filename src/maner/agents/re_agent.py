from __future__ import annotations

from typing import Any

from maner.core.alignment import align_substring_offsets
from maner.core.prompting import PromptManager
from maner.core.schema import SchemaDefinition
from maner.core.types import (
    CandidateSet,
    Evidence,
    ExpertConstraints,
    Relation,
    SpanConstraint,
    UsageCost,
    is_strict_substring,
    is_valid_offsets,
)
from maner.llm.client import LLMClient


class REAgent:
    def __init__(self, llm_client: LLMClient, prompt_manager: PromptManager):
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager

    def run(
        self,
        text: str,
        candidate_set: CandidateSet,
        schema: SchemaDefinition,
        memory_items: list[dict[str, Any]] | None = None,
        allow_span_proposals: bool = False,
    ) -> tuple[list[Relation], ExpertConstraints, UsageCost, dict[str, Any]]:
        memory_items = memory_items or []
        relation_mode = "schema_bound" if schema.relation_constraints else "structure_only"
        augmentation_policy = (
            "enabled: you may propose additional recall spans via `new_spans`."
            if allow_span_proposals
            else "disabled: do not output `new_spans`."
        )
        system, user = self.prompt_manager.render(
            "re_agent",
            text=text,
            candidate_spans=self._candidate_payload(candidate_set),
            memory_items=memory_items,
            entity_types=schema.to_prompt_block(),
            relation_constraints=schema.relation_constraints,
            relation_mode=relation_mode,
            augmentation_policy=augmentation_policy,
        )

        context = None
        if self.llm_client.provider == "mock":
            context = {
                "mock_result": self._heuristic_relations(
                    text=text,
                    candidate_set=candidate_set,
                    relation_constraints=schema.relation_constraints,
                )
            }

        llm_result = self.llm_client.chat_json(
            system_prompt=system,
            user_prompt=user,
            task="re_agent",
            context=context,
        )
        payload = llm_result.parsed_json
        if payload.get("mock") and "result" in payload:
            payload = payload["result"]

        relations = self._parse_relations(
            payload=payload,
            text=text,
            candidate_set=candidate_set,
            relation_constraints=schema.relation_constraints,
        )
        relations = self._apply_coordination_policy(
            text=text,
            candidate_set=candidate_set,
            relations=relations,
            relation_constraints=schema.relation_constraints,
        )
        valid_types = set(schema.entity_type_names)
        structure_support = self._parse_structure_support(
            payload=payload,
            text=text,
            candidate_set=candidate_set,
            valid_types=valid_types,
        )
        span_proposals = (
            self._parse_span_proposals(payload, text, source="re", valid_types=valid_types)
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
            "agent": "re",
            "relation_mode": relation_mode,
            "raw": llm_result.content,
            "parsed": payload,
            "span_proposals": span_proposals,
            "structure_support_count": len(structure_support.per_span),
        }
        return relations, structure_support, cost, trace

    def _candidate_payload(self, candidate_set: CandidateSet) -> list[dict[str, Any]]:
        return [
            {"span_id": sid, "text": span.text, "start": span.start, "end": span.end}
            for sid, span in candidate_set.spans.items()
        ]

    def _heuristic_relations(
        self,
        text: str,
        candidate_set: CandidateSet,
        relation_constraints: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {"relations": [], "new_spans": [], "per_span": {}}

    def _parse_structure_support(
        self,
        payload: dict[str, Any],
        text: str,
        candidate_set: CandidateSet,
        valid_types: set[str],
    ) -> ExpertConstraints:
        raw_per_span = payload.get("per_span", {}) or {}
        constraints = ExpertConstraints(per_span={})
        if not isinstance(raw_per_span, dict):
            return constraints

        for span_id, raw in raw_per_span.items():
            if span_id not in candidate_set.spans or not isinstance(raw, dict):
                continue

            evidence: list[Evidence] = []
            for item in raw.get("evidence", []) or []:
                if not isinstance(item, dict):
                    continue
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
                        Evidence(
                            quote=text[aligned_start:aligned_end],
                            start=aligned_start,
                            end=aligned_end,
                        )
                    )

            candidate_types = _normalize_type_hints(raw.get("candidate_types", []), valid_types)
            excluded_types = _normalize_type_hints(raw.get("excluded_types", []), valid_types)
            confidence = float(raw.get("confidence", 0.0))
            if not evidence:
                confidence = min(confidence, 0.2)

            if not candidate_types and not excluded_types and not evidence:
                continue

            constraints.per_span[str(span_id)] = SpanConstraint(
                candidate_types=candidate_types,
                excluded_types=excluded_types,
                evidence=evidence,
                confidence=confidence,
                rationale=str(raw.get("rationale", "")),
            )
        return constraints

    def _parse_relations(
        self,
        payload: dict[str, Any],
        text: str,
        candidate_set: CandidateSet,
        relation_constraints: list[dict[str, Any]] | None = None,
    ) -> list[Relation]:
        if not relation_constraints:
            return []
        out: list[Relation] = []
        allowed_types = _allowed_relation_types(relation_constraints)
        for item in payload.get("relations", []) or []:
            head_id = str(item.get("head_span_id", ""))
            tail_id = str(item.get("tail_span_id", ""))
            rel_type = str(item.get("rel_type", ""))
            if head_id not in candidate_set.spans or tail_id not in candidate_set.spans:
                continue
            if allowed_types is not None and rel_type not in allowed_types:
                continue

            evidences: list[Evidence] = []
            for ev in item.get("evidence", []) or []:
                quote = str(ev.get("quote", ""))
                start = int(ev.get("start", -1))
                end = int(ev.get("end", -1))
                if is_strict_substring(text, quote, start, end):
                    evidences.append(Evidence(quote=quote, start=start, end=end))
                    continue
                aligned = align_substring_offsets(
                    text=text,
                    quote=quote,
                    start_hint=start,
                    end_hint=end,
                )
                if aligned is not None:
                    aligned_start, aligned_end = aligned
                    evidences.append(
                        Evidence(
                            quote=text[aligned_start:aligned_end],
                            start=aligned_start,
                            end=aligned_end,
                        )
                    )

            conf = float(item.get("confidence", 0.0))
            if not evidences:
                conf = min(conf, 0.2)

            out.append(
                Relation(
                    head_span_id=head_id,
                    rel_type=rel_type,
                    tail_span_id=tail_id,
                    confidence=conf,
                    evidence=evidences,
                )
            )
        return out

    def _apply_coordination_policy(
        self,
        text: str,
        candidate_set: CandidateSet,
        relations: list[Relation],
        relation_constraints: list[dict[str, Any]] | None = None,
    ) -> list[Relation]:
        events = self._find_coordination_events(text, candidate_set)
        if not events:
            return relations

        allowed_types = _allowed_relation_types(relation_constraints)
        drop_regulate_pairs = {frozenset((x, y)) for x, y, _, _ in events}

        kept: list[Relation] = []
        for r in relations:
            rel_l = r.rel_type.lower()
            if "regulat" in rel_l and frozenset((r.head_span_id, r.tail_span_id)) in drop_regulate_pairs:
                # For "X and Y regulate/cooperate in regulating Z", avoid defaulting to X regulates Y.
                continue
            kept.append(r)

        enforced: list[Relation] = []
        for x_id, y_id, z_id, has_cooperate in events:
            x = candidate_set.spans[x_id]
            z = candidate_set.spans[z_id]
            ev_start = x.start
            ev_end = z.end
            evidence = [
                Evidence(
                    quote=text[ev_start:ev_end],
                    start=ev_start,
                    end=ev_end,
                )
            ]

            if has_cooperate and _relation_allowed("cooperates_with", allowed_types):
                enforced.append(
                    Relation(
                        head_span_id=x_id,
                        rel_type="cooperates_with",
                        tail_span_id=y_id,
                        confidence=0.78,
                        evidence=evidence,
                    )
                )

            if _relation_allowed("regulates", allowed_types):
                enforced.append(
                    Relation(
                        head_span_id=x_id,
                        rel_type="regulates",
                        tail_span_id=z_id,
                        confidence=0.74,
                        evidence=evidence,
                    )
                )
                enforced.append(
                    Relation(
                        head_span_id=y_id,
                        rel_type="regulates",
                        tail_span_id=z_id,
                        confidence=0.74,
                        evidence=evidence,
                    )
                )

        return _dedup_relations(kept + enforced)

    def _find_coordination_events(
        self,
        text: str,
        candidate_set: CandidateSet,
    ) -> list[tuple[str, str, str, bool]]:
        spans = sorted(candidate_set.spans.items(), key=lambda x: (x[1].start, x[1].end))
        events: list[tuple[str, str, str, bool]] = []

        for i in range(len(spans)):
            x_id, x_span = spans[i]
            for j in range(i + 1, len(spans)):
                y_id, y_span = spans[j]
                between_xy = text[x_span.end : y_span.start].lower()
                if "and" not in between_xy:
                    continue
                for k in range(j + 1, len(spans)):
                    z_id, z_span = spans[k]
                    if z_span.end - x_span.start > 200:
                        break
                    between_yz = text[y_span.end : z_span.start].lower()
                    if "regulat" not in between_yz:
                        continue
                    has_cooperate = "cooperat" in between_yz
                    events.append((x_id, y_id, z_id, has_cooperate))
        return events

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
            aligned = (
                align_substring_offsets(
                    text=text,
                    quote=span_text,
                    start_hint=start,
                    end_hint=end,
                )
                if span_text
                else None
            )
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


def _allowed_relation_types(relation_constraints: list[dict[str, Any]] | None) -> set[str] | None:
    if not relation_constraints:
        return None
    names = {str(item.get("name", "")) for item in relation_constraints if isinstance(item, dict)}
    names = {name for name in names if name}
    return names or None


def _relation_allowed(rel_type: str, allowed_types: set[str] | None) -> bool:
    if allowed_types is None:
        return True
    return rel_type in allowed_types


def _dedup_relations(relations: list[Relation]) -> list[Relation]:
    best: dict[tuple[str, str, str], Relation] = {}
    for rel in relations:
        key = (rel.head_span_id, rel.rel_type, rel.tail_span_id)
        prev = best.get(key)
        if prev is None or rel.confidence > prev.confidence:
            best[key] = rel
    return list(best.values())


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
