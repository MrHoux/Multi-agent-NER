from __future__ import annotations

from collections import defaultdict
from typing import Any

from maner.core.alignment import align_substring_offsets
from maner.core.prompting import PromptManager
from maner.core.schema import SchemaDefinition
from maner.core.types import CandidateSet, UsageCost, is_strict_substring
from maner.llm.client import LLMClient
from maner.retrieval import NullRetriever, WikipediaRetriever


class RAGAgent:
    def __init__(
        self,
        llm_client: LLMClient,
        prompt_manager: PromptManager,
        settings: dict[str, Any] | None = None,
        retriever: Any | None = None,
    ):
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager
        self.settings = dict(settings or {})
        self.max_open_questions = int(self.settings.get("max_open_questions", 20))
        if retriever is not None:
            self.retriever = retriever
        elif self.llm_client.provider == "mock":
            self.retriever = NullRetriever()
        else:
            self.retriever = WikipediaRetriever(self.settings)

    def run(
        self,
        text: str,
        candidate_set: CandidateSet,
        schema: SchemaDefinition,
        memory_items: list[dict[str, Any]] | None = None,
        expert_retrieval_plan: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], UsageCost, dict[str, Any]]:
        memory_items = memory_items or []
        expert_retrieval_plan = expert_retrieval_plan or {}
        query_plan, query_cost, query_trace = self._plan_queries(
            text=text,
            candidate_set=candidate_set,
            schema=schema,
            memory_items=memory_items,
            expert_retrieval_plan=expert_retrieval_plan,
        )
        retrieved_documents, retrieval_trace = self.retriever.retrieve(query_plan)

        synth_payload, synth_cost, synth_trace = self._run_synth_stage(
            text=text,
            candidate_set=candidate_set,
            schema=schema,
            memory_items=memory_items,
            query_plan=query_plan,
            retrieved_documents=retrieved_documents,
        )

        handoff = self._parse_handoff(synth_payload, text, candidate_set, schema)
        cost = UsageCost(
            calls=query_cost.calls + synth_cost.calls,
            prompt_tokens=_sum_opt(query_cost.prompt_tokens, synth_cost.prompt_tokens),
            completion_tokens=_sum_opt(query_cost.completion_tokens, synth_cost.completion_tokens),
            total_tokens=_sum_opt(query_cost.total_tokens, synth_cost.total_tokens),
            latency_ms=(query_cost.latency_ms or []) + (synth_cost.latency_ms or []),
        )
        trace = {
            "agent": "rag",
            "query_stage": query_trace,
            "retrieval": retrieval_trace,
            "synth_stage": synth_trace,
            "query_plan": query_plan,
        }
        return handoff, cost, trace

    def _plan_queries(
        self,
        text: str,
        candidate_set: CandidateSet,
        schema: SchemaDefinition,
        memory_items: list[dict[str, Any]],
        expert_retrieval_plan: dict[str, Any],
    ) -> tuple[dict[str, Any], UsageCost, dict[str, Any]]:
        retrieval_requests = expert_retrieval_plan.get("retrieval_requests", []) or []
        if retrieval_requests:
            query_plan = self._build_query_plan_from_expert_requests(
                candidate_set=candidate_set,
                expert_retrieval_plan=expert_retrieval_plan,
            )
            return query_plan, UsageCost(), {"mode": "expert_guided", "parsed": query_plan}
        return self._run_query_stage(
            text=text,
            candidate_set=candidate_set,
            schema=schema,
            memory_items=memory_items,
            expert_retrieval_plan=expert_retrieval_plan,
        )

    def _run_query_stage(
        self,
        text: str,
        candidate_set: CandidateSet,
        schema: SchemaDefinition,
        memory_items: list[dict[str, Any]],
        expert_retrieval_plan: dict[str, Any],
    ) -> tuple[dict[str, Any], UsageCost, dict[str, Any]]:
        system, user = self.prompt_manager.render(
            "rag_query_agent",
            text=text,
            candidate_spans=self._candidate_prompt_payload(candidate_set),
            memory_items=memory_items,
            expert_retrieval_plan=expert_retrieval_plan,
            entity_types=schema.to_prompt_block(),
        )

        context = None
        if self.llm_client.provider == "mock":
            context = {
                "mock_result": self._heuristic_query_plan(
                    text=text,
                    candidate_set=candidate_set,
                    memory_items=memory_items,
                    expert_retrieval_plan=expert_retrieval_plan,
                )
            }

        llm_result = self.llm_client.chat_json(
            system_prompt=system,
            user_prompt=user,
            task="rag_query_agent",
            context=context,
        )
        payload = llm_result.parsed_json
        if payload.get("mock") and "result" in payload:
            payload = payload["result"]

        query_plan = self._parse_query_plan(payload, candidate_set)
        cost = UsageCost(
            calls=llm_result.usage.calls,
            prompt_tokens=llm_result.usage.prompt_tokens,
            completion_tokens=llm_result.usage.completion_tokens,
            total_tokens=llm_result.usage.total_tokens,
            latency_ms=llm_result.usage.latency_ms or [],
        )
        trace = {"mode": "llm", "raw": llm_result.content, "parsed": payload}
        return query_plan, cost, trace

    def _run_synth_stage(
        self,
        text: str,
        candidate_set: CandidateSet,
        schema: SchemaDefinition,
        memory_items: list[dict[str, Any]],
        query_plan: dict[str, Any],
        retrieved_documents: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], UsageCost, dict[str, Any]]:
        system, user = self.prompt_manager.render(
            "rag_synth_agent",
            text=text,
            candidate_spans=self._candidate_prompt_payload(candidate_set),
            memory_items=memory_items,
            query_plan=query_plan,
            retrieved_documents=retrieved_documents,
            entity_types=schema.to_prompt_block(),
        )

        context = None
        if self.llm_client.provider == "mock":
            context = {
                "mock_result": self._heuristic_handoff(
                    text=text,
                    candidate_set=candidate_set,
                    schema=schema,
                    memory_items=memory_items,
                    query_plan=query_plan,
                    retrieved_documents=retrieved_documents,
                )
            }

        llm_result = self.llm_client.chat_json(
            system_prompt=system,
            user_prompt=user,
            task="rag_synth_agent",
            context=context,
        )
        payload = llm_result.parsed_json
        if payload.get("mock") and "result" in payload:
            payload = payload["result"]

        cost = UsageCost(
            calls=llm_result.usage.calls,
            prompt_tokens=llm_result.usage.prompt_tokens,
            completion_tokens=llm_result.usage.completion_tokens,
            total_tokens=llm_result.usage.total_tokens,
            latency_ms=llm_result.usage.latency_ms or [],
        )
        trace = {"raw": llm_result.content, "parsed": payload}
        return payload, cost, trace

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

    def _build_query_plan_from_expert_requests(
        self,
        candidate_set: CandidateSet,
        expert_retrieval_plan: dict[str, Any],
    ) -> dict[str, Any]:
        candidate_queries: list[dict[str, Any]] = []
        for idx, item in enumerate(expert_retrieval_plan.get("retrieval_requests", []) or [], start=1):
            if not isinstance(item, dict):
                continue
            span_ids = [
                str(span_id)
                for span_id in item.get("span_ids", []) or []
                if str(span_id) in candidate_set.spans
            ]
            if not span_ids:
                continue
            span_texts = [candidate_set.spans[span_id].text for span_id in span_ids[:2]]
            question = str(item.get("question", "")).strip()
            if not question:
                continue
            query_text = question
            if span_texts:
                query_text = f"{question} Context mention: {' / '.join(span_texts)}."
            candidate_queries.append(
                {
                    "q_id": str(item.get("request_id", f"rq_{idx:03d}")),
                    "span_ids": span_ids,
                    "query": query_text,
                    "focus": "typing",
                    "priority": str(item.get("priority", "medium")),
                }
            )

        return {
            "query_plan_id": "rag_query_plan_expert_guided",
            "candidate_queries": candidate_queries[: max(1, self.max_open_questions)],
            "selected_memory_ids": [],
            "priority_notes": [str(x) for x in expert_retrieval_plan.get("global_notes", []) or []],
            "rationale": str(expert_retrieval_plan.get("rationale", "")),
        }

    def _heuristic_handoff(
        self,
        text: str,
        candidate_set: CandidateSet,
        schema: SchemaDefinition,
        memory_items: list[dict[str, Any]],
        query_plan: dict[str, Any],
        retrieved_documents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        valid_types = set(schema.entity_type_names)
        retrieved_facts: list[dict[str, Any]] = []
        per_span_hints: dict[str, Any] = {}

        for idx, item in enumerate(memory_items, start=1):
            linked_span_ids = []
            key = str(item.get("key", ""))
            for sid, span in candidate_set.spans.items():
                if key and key.lower() in span.text.lower():
                    linked_span_ids.append(sid)

            retrieved_facts.append(
                {
                    "fact_id": f"fact_mem_{idx:03d}",
                    "source": "memory",
                    "content": str(item.get("value", {})),
                    "linked_span_ids": linked_span_ids,
                    "relevance": float(item.get("score", item.get("confidence", 0.0))),
                }
            )

            item_value = item.get("value", {}) if isinstance(item.get("value", {}), dict) else {}
            ent_type = str(item_value.get("ent_type", ""))
            if ent_type not in valid_types:
                continue
            for sid in linked_span_ids:
                if sid not in candidate_set.spans:
                    continue
                span = candidate_set.spans[sid]
                per_span_hints[sid] = {
                    "candidate_types": [ent_type],
                    "evidence": [{"quote": text[span.start : span.end], "start": span.start, "end": span.end}],
                    "query_note": "memory_linked_type_hint",
                    "sources": ["memory"],
                }

        for idx, doc in enumerate(retrieved_documents, start=1):
            linked = [sid for sid in doc.get("linked_span_ids", []) if sid in candidate_set.spans]
            retrieved_facts.append(
                {
                    "fact_id": f"fact_wiki_{idx:03d}",
                    "source": "wikipedia",
                    "content": str(doc.get("summary", "")),
                    "linked_span_ids": linked,
                    "relevance": float(doc.get("relevance", 0.0)),
                }
            )

        open_questions: list[dict[str, Any]] = []
        raw_queries = query_plan.get("candidate_queries", []) or []
        for idx, q in enumerate(raw_queries, start=1):
            if not isinstance(q, dict):
                continue
            q_span_ids = [sid for sid in q.get("span_ids", []) if sid in candidate_set.spans]
            if not q_span_ids:
                continue
            open_questions.append(
                {
                    "q_id": str(q.get("q_id", f"q_{idx:03d}")),
                    "span_id": q_span_ids[0],
                    "question": str(q.get("query", "")),
                    "priority": str(q.get("priority", "medium")),
                }
            )

        return {
            "handoff_id": "rag_handoff_001",
            "retrieved_facts": retrieved_facts,
            "per_span_hints": per_span_hints,
            "open_questions": open_questions[: self.max_open_questions],
            "priority_notes": ["Prefer evidence-grounded hints", "Do not force unsupported labels"],
            "rationale": "memory and wikipedia evidence synthesis",
        }

    def _heuristic_query_plan(
        self,
        text: str,
        candidate_set: CandidateSet,
        memory_items: list[dict[str, Any]],
        expert_retrieval_plan: dict[str, Any],
    ) -> dict[str, Any]:
        if expert_retrieval_plan.get("retrieval_requests"):
            return self._build_query_plan_from_expert_requests(candidate_set, expert_retrieval_plan)

        candidate_queries = []
        selected_memory_ids: list[str] = []
        lower_text = text.lower()
        for idx, (sid, span) in enumerate(candidate_set.spans.items(), start=1):
            priority = "high" if span.text and span.text[0].isupper() else "medium"
            candidate_queries.append(
                {
                    "q_id": f"rq_{idx:03d}",
                    "span_ids": [sid],
                    "query": "Clarify the identity or entity type of this mention using external reference knowledge only when local context is insufficient.",
                    "focus": "typing",
                    "priority": priority,
                }
            )

        for idx, item in enumerate(memory_items, start=1):
            key = str(item.get("key", "")).lower()
            if key and key in lower_text:
                selected_memory_ids.append(f"mem_{idx:03d}")

        return {
            "query_plan_id": "rag_query_plan_001",
            "candidate_queries": candidate_queries[:30],
            "selected_memory_ids": selected_memory_ids,
            "priority_notes": ["Favor high-relevance memory", "Ask explicit disambiguation questions"],
            "rationale": "query planning before synthesis",
        }

    def _parse_query_plan(self, payload: dict[str, Any], candidate_set: CandidateSet) -> dict[str, Any]:
        queries = []
        raw_queries = payload.get("candidate_queries", []) or []
        for q in raw_queries:
            if not isinstance(q, dict):
                continue
            span_ids = [sid for sid in q.get("span_ids", []) if sid in candidate_set.spans]
            if not span_ids:
                continue
            queries.append(
                {
                    "q_id": str(q.get("q_id", "")),
                    "span_ids": span_ids,
                    "query": str(q.get("query", "")),
                    "focus": str(q.get("focus", "typing")),
                    "priority": str(q.get("priority", "medium")),
                }
            )

        return {
            "query_plan_id": str(payload.get("query_plan_id", "rag_query_plan")),
            "candidate_queries": queries,
            "selected_memory_ids": [str(x) for x in payload.get("selected_memory_ids", []) or []],
            "priority_notes": [str(x) for x in payload.get("priority_notes", []) or []],
            "rationale": str(payload.get("rationale", "")),
        }

    def _parse_handoff(
        self,
        payload: dict[str, Any],
        text: str,
        candidate_set: CandidateSet,
        schema: SchemaDefinition,
    ) -> dict[str, Any]:
        valid_types = set(schema.entity_type_names)
        handoff_id = str(payload.get("handoff_id", "rag_handoff"))
        raw_facts = payload.get("retrieved_facts", []) or []
        raw_hints = payload.get("per_span_hints", {}) or {}
        raw_questions = payload.get("open_questions", []) or []

        retrieved_facts: list[dict[str, Any]] = []
        fact_sources_by_span: dict[str, set[str]] = defaultdict(set)
        for item in raw_facts:
            if not isinstance(item, dict):
                continue
            fact_id = str(item.get("fact_id", ""))
            source = str(item.get("source", "external"))
            content = str(item.get("content", ""))
            linked = [sid for sid in item.get("linked_span_ids", []) if sid in candidate_set.spans]
            retrieved_facts.append(
                {
                    "fact_id": fact_id,
                    "source": source,
                    "content": content,
                    "linked_span_ids": linked,
                    "relevance": float(item.get("relevance", 0.0)),
                }
            )
            for sid in linked:
                fact_sources_by_span[sid].add(source)

        per_span_hints: dict[str, Any] = {}
        for sid, hint in raw_hints.items():
            if sid not in candidate_set.spans or not isinstance(hint, dict):
                continue
            candidate_types = [
                t for t in hint.get("candidate_types", []) if isinstance(t, str) and t in valid_types
            ]
            evidences = []
            for ev in hint.get("evidence", []) or []:
                if not isinstance(ev, dict):
                    continue
                quote = str(ev.get("quote", ""))
                start = int(ev.get("start", -1))
                end = int(ev.get("end", -1))
                if is_strict_substring(text, quote, start, end):
                    evidences.append({"quote": quote, "start": start, "end": end})
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
                        {
                            "quote": text[aligned_start:aligned_end],
                            "start": aligned_start,
                            "end": aligned_end,
                        }
                    )
            sources = [
                str(x)
                for x in hint.get("sources", []) or []
                if isinstance(x, str) and str(x).strip()
            ]
            if not sources:
                sources = sorted(fact_sources_by_span.get(sid, set()))
            if not sources:
                sources = ["llm_inference"]
            per_span_hints[sid] = {
                "candidate_types": candidate_types,
                "evidence": evidences,
                "query_note": str(hint.get("query_note", "")),
                "sources": sources,
            }

        open_questions: list[dict[str, Any]] = []
        for q in raw_questions:
            if not isinstance(q, dict):
                continue
            sid = str(q.get("span_id", ""))
            if sid and sid not in candidate_set.spans:
                continue
            open_questions.append(
                {
                    "q_id": str(q.get("q_id", "")),
                    "span_id": sid,
                    "question": str(q.get("question", "")),
                    "priority": str(q.get("priority", "medium")),
                }
            )

        return {
            "handoff_id": handoff_id,
            "retrieved_facts": retrieved_facts,
            "per_span_hints": per_span_hints,
            "open_questions": open_questions[: self.max_open_questions],
            "priority_notes": list(payload.get("priority_notes", []) or []),
            "rationale": str(payload.get("rationale", "")),
        }


def _sum_opt(a: int | None, b: int | None) -> int | None:
    if a is None and b is None:
        return None
    return int(a or 0) + int(b or 0)
