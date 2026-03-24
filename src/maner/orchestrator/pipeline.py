from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import re
import threading
import time
from pathlib import Path
from typing import Any

from maner.agents.adjudicator_agent import AdjudicatorAgent
from maner.agents.candidate_agent import CandidateAgent
from maner.agents.debate_protocol import DebateProtocol
from maner.agents.disambiguation_agent import DisambiguationAgent
from maner.agents.expert_agent import ExpertAgent
from maner.agents.ner_agent import NERAgent
from maner.agents.rag_agent import RAGAgent
from maner.agents.re_agent import REAgent
from maner.agents.verifier import Verifier
from maner.core.dataset import build_reader
from maner.core.prompting import PromptManager
from maner.core.schema import load_schema
from maner.core.types import (
    CandidateSet,
    ConflictCluster,
    Evidence,
    ExpertConstraints,
    Mention,
    NERHypothesis,
    Relation,
    Span,
    SpanConstraint,
    UsageCost,
    is_valid_offsets,
    span_iou,
    to_dict,
)
from maner.llm.client import LLMClient
from maner.memory.store import MemoryStore
from maner.orchestrator.conflict import apply_risk_levels, build_conflict_clusters
from maner.orchestrator.triage import triage_conflicts


class PipelineRunner:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.prompts = PromptManager(self._prompt_sources(config))
        self.llm = LLMClient(config.get("llm", {}))

        data_cfg = config.get("data", {})
        self.schema = load_schema(data_cfg["schema_path"])
        self.reader = build_reader(
            data_path=data_cfg["data_path"],
            reader_type=data_cfg.get("reader_type", "generic_jsonl"),
        )

        self.ablations = config.get("ablations", {})
        self.pipeline_cfg = config.get("pipeline", {})
        self.memory_cfg = config.get("memory", {})
        self.verifier_cfg = config.get("verifier", {})
        self.progress_logging = bool(self.pipeline_cfg.get("progress_logging", False))
        self.progress_agent_trace = bool(
            self.pipeline_cfg.get("progress_agent_trace", False)
        )
        self._log_lock = threading.Lock()

        self.candidate_agent = CandidateAgent(
            self.llm,
            self.prompts,
            settings=self.pipeline_cfg.get("candidate", {}),
        )
        self.rag_agent = RAGAgent(self.llm, self.prompts, settings=config.get("rag", {}))
        self.expert_agent = ExpertAgent(self.llm, self.prompts)
        self.re_agent = REAgent(self.llm, self.prompts)
        self.ner_agent = NERAgent(self.llm, self.prompts)
        self.adjudicator = AdjudicatorAgent(self.llm, self.prompts)
        self.debate_protocol = DebateProtocol(
            self.llm,
            self.prompts,
            max_turns=int(self.pipeline_cfg.get("max_debate_turns", 5)),
            epsilon=float(self.pipeline_cfg.get("early_stop_delta", 0.01)),
        )
        self.disambiguation_agent = DisambiguationAgent(self.llm, self.prompts)
        self.verifier = Verifier(
            self.llm,
            self.prompts,
            use_llm=bool(self.verifier_cfg.get("use_llm", False)),
            strict_drop_invalid=bool(self.verifier_cfg.get("strict_drop_invalid", True)),
        )

        self.memory_store = None
        memory_enabled = bool(self.memory_cfg.get("enabled", True)) and not bool(
            self.ablations.get("w_o_memory", False)
        )
        if memory_enabled:
            self.memory_store = MemoryStore(self.memory_cfg.get("sqlite_path", "outputs/memory.db"))

    def _prompt_sources(self, config: dict[str, Any]) -> list[Path]:
        repo_root = Path(__file__).resolve().parents[3]
        sources = [repo_root / "configs" / "prompts_cot.yaml"]

        configured = config.get("prompts_path")
        if configured:
            if isinstance(configured, list):
                sources.extend(_resolve_prompt_path(repo_root, item) for item in configured)
            else:
                sources.append(_resolve_prompt_path(repo_root, configured))

        overlays = config.get("prompt_overlays", [])
        if isinstance(overlays, list):
            sources.extend(_resolve_prompt_path(repo_root, item) for item in overlays)

        deduped: list[Path] = []
        seen: set[Path] = set()
        for path in sources:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            deduped.append(resolved)
        return deduped

    def close(self) -> None:
        if self.memory_store is not None:
            self.memory_store.close()

    def _progress(self, event: str, **fields: Any) -> None:
        if not self.progress_logging:
            return
        parts = [f"{key}={self._format_log_value(fields[key])}" for key in sorted(fields)]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"[{timestamp}] [pipeline] {event}"
        if parts:
            message += " | " + " ".join(parts)
        with self._log_lock:
            print(message, flush=True)

    @staticmethod
    def _format_log_value(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.3f}"
        if isinstance(value, (dict, list, tuple, set)):
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        return str(value)

    def _agent_progress(self, agent: str, stage: str, **fields: Any) -> None:
        if not self.progress_agent_trace:
            return
        self._progress(f"{agent}_{stage}", **fields)

    def _run_expert_branch(
        self,
        *,
        sample_id: str,
        text: str,
        candidate_set: CandidateSet,
        memory_items: list[dict[str, Any]],
        use_expert: bool,
        use_rag: bool,
        allow_span_proposals: bool,
    ) -> dict[str, Any]:
        branch_cost = UsageCost()
        branch_traces: dict[str, Any] = {}
        communications: list[dict[str, Any]] = []
        expert_retrieval_plan: dict[str, Any] = {}
        rag_handoff: dict[str, Any] = {}
        constraints = ExpertConstraints()

        if use_rag and use_expert and candidate_set.spans:
            started = time.perf_counter()
            self._agent_progress("expert_retrieval_agent", "start", sample_id=sample_id)
            expert_retrieval_plan, cost, trace = self.expert_agent.plan_retrieval(
                text=text,
                candidate_set=candidate_set,
                schema=self.schema,
                memory_items=memory_items,
            )
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            _accumulate_cost(branch_cost, cost)
            branch_traces["expert_retrieval"] = trace
            self._agent_progress(
                "expert_retrieval_agent",
                "done",
                sample_id=sample_id,
                llm_calls=cost.calls,
                retrieval_requests=len(expert_retrieval_plan.get("retrieval_requests", []) or []),
                elapsed_ms=elapsed_ms,
            )
            communications.append(
                {
                    "from": "expert_agent",
                    "to": "rag_agent",
                    "message_type": "retrieval_requests",
                    "request_count": len(expert_retrieval_plan.get("retrieval_requests", []) or []),
                }
            )

            started = time.perf_counter()
            self._agent_progress("rag_agent", "start", sample_id=sample_id)
            rag_handoff, cost, trace = self.rag_agent.run(
                text=text,
                candidate_set=candidate_set,
                schema=self.schema,
                memory_items=memory_items,
                expert_retrieval_plan=expert_retrieval_plan,
            )
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            _accumulate_cost(branch_cost, cost)
            branch_traces["rag"] = trace
            self._agent_progress(
                "rag_agent",
                "done",
                sample_id=sample_id,
                llm_calls=cost.calls,
                hinted_spans=len((rag_handoff.get("per_span_hints", {}) or {})),
                elapsed_ms=elapsed_ms,
            )
            hinted = rag_handoff.get("per_span_hints", {}) or {}
            hinted_count = len(hinted) if isinstance(hinted, dict) else 0
            communications.append(
                {
                    "from": "rag_agent",
                    "to": "expert_agent",
                    "message_type": "rag_handoff",
                    "handoff_id": str(rag_handoff.get("handoff_id", "")),
                    "hinted_spans": hinted_count,
                    "open_questions": len(rag_handoff.get("open_questions", []) or []),
                }
            )
        elif use_rag and use_expert and not candidate_set.spans:
            branch_traces["expert_retrieval"] = {
                "skipped": True,
                "reason": "no_candidate_spans",
            }
            branch_traces["rag"] = {
                "skipped": True,
                "reason": "no_candidate_spans",
            }
        else:
            branch_traces["expert_retrieval"] = {"disabled": True}
            branch_traces["rag"] = {"disabled": True}

        if use_expert:
            started = time.perf_counter()
            self._agent_progress("expert_agent", "start", sample_id=sample_id)
            constraints, cost, trace = self.expert_agent.run(
                text=text,
                candidate_set=candidate_set,
                schema=self.schema,
                memory_items=memory_items,
                rag_handoff=rag_handoff,
                allow_span_proposals=allow_span_proposals,
            )
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            _accumulate_cost(branch_cost, cost)
            branch_traces["expert"] = trace
            self._agent_progress(
                "expert_agent",
                "done",
                sample_id=sample_id,
                llm_calls=cost.calls,
                constrained_spans=len(constraints.per_span),
                elapsed_ms=elapsed_ms,
            )
            communications.append(
                {
                    "from": "expert_agent",
                    "to": "pipeline",
                    "message_type": "expert_constraints",
                    "constrained_spans": len(constraints.per_span),
                    "span_proposals": len(trace.get("span_proposals", []) or []),
                }
            )
            if use_rag and branch_traces.get("rag", {}).get("disabled") is not True:
                branch_traces["rag_expert_alignment"] = _summarize_rag_expert_alignment(
                    rag_handoff=rag_handoff,
                    constraints=constraints,
                )
        else:
            branch_traces["expert"] = {"disabled": True}

        return {
            "cost": branch_cost,
            "traces": branch_traces,
            "communications": communications,
            "expert_retrieval_plan": expert_retrieval_plan,
            "rag_handoff": rag_handoff,
            "constraints": constraints,
        }

    def _run_re_branch(
        self,
        *,
        sample_id: str,
        text: str,
        candidate_set: CandidateSet,
        memory_items: list[dict[str, Any]],
        relation_schema_present: bool,
        allow_span_proposals: bool,
    ) -> dict[str, Any]:
        branch_cost = UsageCost()
        branch_traces: dict[str, Any] = {}
        communications: list[dict[str, Any]] = []
        relations: list[Relation] = []
        re_structure_support = ExpertConstraints()

        re_role_name = "re_agent" if relation_schema_present else "in_context_agent"
        started = time.perf_counter()
        self._agent_progress(re_role_name, "start", sample_id=sample_id)
        relations, re_structure_support, cost, trace = self.re_agent.run(
            text,
            candidate_set,
            self.schema,
            memory_items,
            allow_span_proposals=allow_span_proposals,
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        _accumulate_cost(branch_cost, cost)
        branch_traces["re"] = trace
        self._agent_progress(
            re_role_name,
            "done",
            sample_id=sample_id,
            llm_calls=cost.calls,
            relations=len(relations),
            elapsed_ms=elapsed_ms,
        )
        communications.append(
            {
                "from": re_role_name,
                "to": "pipeline",
                "message_type": "relations_and_proposals"
                if relation_schema_present
                else "in_context_support",
                "relations": len(relations),
                "span_proposals": len(trace.get("span_proposals", []) or []),
                "structure_supported_spans": len(re_structure_support.per_span),
            }
        )

        return {
            "cost": branch_cost,
            "traces": branch_traces,
            "communications": communications,
            "relations": relations,
            "re_structure_support": re_structure_support,
        }

    def _run_expert_ner_branch(
        self,
        *,
        sample_id: str,
        text: str,
        use_expert: bool,
        candidate_set: CandidateSet,
        constraints: ExpertConstraints,
        pre_augmentation_candidate_set: CandidateSet,
        rerun_constraints_for_added: ExpertConstraints,
        added_ids: list[str],
        rerun_after_aug: bool,
    ) -> dict[str, Any]:
        branch_cost = UsageCost()
        if not use_expert:
            return {
                "cost": branch_cost,
                "trace": {"disabled": True},
                "hypothesis": NERHypothesis(mentions=[], source="expert"),
                "llm_calls": 0,
            }
        localize_augmented_ner = bool(
            self.pipeline_cfg.get("localize_augmented_ner_inference", True)
        )
        should_localize_augmented_ner = bool(
            localize_augmented_ner and added_ids and rerun_after_aug
        )

        started = time.perf_counter()
        self._agent_progress("ner_agent_expert", "start", sample_id=sample_id)
        if should_localize_augmented_ner:
            added_id_set = set(added_ids)
            base_candidate_subset = pre_augmentation_candidate_set
            added_candidate_subset = _subset_candidate_set(candidate_set, added_id_set)
            added_constraint_subset = _subset_constraints(
                rerun_constraints_for_added,
                added_id_set,
            )

            y_exp_base, cost_base, trace_base = self.ner_agent.run_with_expert(
                text,
                base_candidate_subset,
                self.schema,
                constraints,
            )
            _accumulate_cost(branch_cost, cost_base)
            y_exp_added, cost_added, trace_added = self.ner_agent.run_with_expert(
                text,
                added_candidate_subset,
                self.schema,
                added_constraint_subset,
            )
            _accumulate_cost(branch_cost, cost_added)
            y_exp = NERHypothesis(
                mentions=_merge_mentions(y_exp_base.mentions + y_exp_added.mentions),
                source="expert",
            )
            trace = {
                "localized_augmented_inference": True,
                "base_trace": trace_base,
                "added_trace": trace_added,
                "base_mentions": len(y_exp_base.mentions),
                "added_mentions": len(y_exp_added.mentions),
                "output_mentions": len(y_exp.mentions),
                "added_span_ids": sorted(added_id_set),
            }
            llm_calls = cost_base.calls + cost_added.calls
        else:
            y_exp, cost, trace = self.ner_agent.run_with_expert(
                text, candidate_set, self.schema, constraints
            )
            _accumulate_cost(branch_cost, cost)
            llm_calls = cost.calls
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        self._agent_progress(
            "ner_agent_expert",
            "done",
            sample_id=sample_id,
            llm_calls=llm_calls,
            mentions=len(y_exp.mentions),
            elapsed_ms=elapsed_ms,
        )
        return {
            "cost": branch_cost,
            "trace": trace,
            "hypothesis": y_exp,
            "llm_calls": llm_calls,
        }

    def _run_secondary_ner_branch(
        self,
        *,
        sample_id: str,
        text: str,
        candidate_set: CandidateSet,
        relation_schema_present: bool,
        relations: list[Relation],
        re_structure_support: ExpertConstraints,
    ) -> dict[str, Any]:
        branch_cost = UsageCost()
        ner_secondary_name = "ner_agent_re" if relation_schema_present else "ner_agent_in_context"
        started = time.perf_counter()
        self._agent_progress(ner_secondary_name, "start", sample_id=sample_id)
        if relation_schema_present:
            y_re, cost, trace = self.ner_agent.run_with_re(
                text, candidate_set, self.schema, relations
            )
        else:
            y_re, cost, trace = self.ner_agent.run_with_context(
                text, candidate_set, self.schema, re_structure_support
            )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        _accumulate_cost(branch_cost, cost)
        self._agent_progress(
            ner_secondary_name,
            "done",
            sample_id=sample_id,
            llm_calls=cost.calls,
            mentions=len(y_re.mentions),
            elapsed_ms=elapsed_ms,
        )
        return {
            "cost": branch_cost,
            "trace": trace,
            "hypothesis": y_re,
        }

    def run(self) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        continue_on_error = bool(self.pipeline_cfg.get("continue_on_error", True))
        sample_max_retries = int(self.pipeline_cfg.get("sample_max_retries", 1))
        sample_retry_backoff_s = float(self.pipeline_cfg.get("sample_retry_backoff_s", 1.0))
        sample_post_error_retries = int(
            self.pipeline_cfg.get("sample_post_error_retries", 0)
        )
        sample_post_error_backoff_s = float(
            self.pipeline_cfg.get("sample_post_error_backoff_s", 5.0)
        )
        for idx, sample in enumerate(self.reader.iter_samples(), start=1):
            sample_started = time.time()
            self._progress(
                "sample_start",
                idx=idx,
                sample_id=sample.sample_id,
                chars=len(sample.text),
            )
            attempts = max(0, sample_max_retries) + 1
            for attempt in range(attempts):
                try:
                    result = self._run_sample(sample.sample_id, sample.text)
                    outputs.append(result)
                    self._progress(
                        "sample_done",
                        idx=idx,
                        sample_id=sample.sample_id,
                        mentions=len(result.get("mentions", [])),
                        calls=result.get("costs", {}).get("calls", 0),
                        wall_s=round(time.time() - sample_started, 3),
                    )
                    break
                except Exception as exc:
                    is_last_attempt = attempt >= attempts - 1
                    if (not is_last_attempt) and _is_retryable_pipeline_exception(exc):
                        self._progress(
                            "sample_retry",
                            idx=idx,
                            sample_id=sample.sample_id,
                            attempt=attempt + 1,
                            error=exc.__class__.__name__,
                        )
                        sleep_s = sample_retry_backoff_s * (2**attempt)
                        if sleep_s > 0:
                            time.sleep(sleep_s)
                        continue
                    recovered = None
                    if continue_on_error and _is_retryable_pipeline_exception(exc):
                        for post_idx in range(max(0, sample_post_error_retries)):
                            sleep_s = sample_post_error_backoff_s * (post_idx + 1)
                            if sleep_s > 0:
                                time.sleep(sleep_s)
                            try:
                                recovered = self._run_sample(sample.sample_id, sample.text)
                                break
                            except Exception as post_exc:
                                exc = post_exc
                    if recovered is not None:
                        outputs.append(recovered)
                        self._progress(
                            "sample_recovered",
                            idx=idx,
                            sample_id=sample.sample_id,
                            mentions=len(recovered.get("mentions", [])),
                            calls=recovered.get("costs", {}).get("calls", 0),
                            wall_s=round(time.time() - sample_started, 3),
                        )
                        break
                    if not continue_on_error:
                        raise
                    self._progress(
                        "sample_failed",
                        idx=idx,
                        sample_id=sample.sample_id,
                        error=exc.__class__.__name__,
                        wall_s=round(time.time() - sample_started, 3),
                    )
                    outputs.append(
                        {
                            "id": sample.sample_id,
                            "text": sample.text,
                            "mentions": [],
                            "traces": {
                                "runtime_error": {
                                    "message": str(exc),
                                    "type": exc.__class__.__name__,
                                }
                            },
                            "costs": {
                                "calls": 0,
                                "prompt_tokens": None,
                                "completion_tokens": None,
                                "total_tokens": None,
                                "latency_ms": [],
                                "debate_turns": 0,
                                "debate_triggered": 0,
                                "debate_trigger_rate": 0.0,
                            },
                        }
                    )
                    break
        return outputs

    def _run_sample(self, sample_id: str, text: str) -> dict[str, Any]:
        total_cost = UsageCost()
        traces: dict[str, Any] = {}
        communications: list[dict[str, Any]] = []
        traces["communications"] = communications

        direct_seed_enabled = bool(self.pipeline_cfg.get("enable_direct_seed_ner", True))
        direct_seed_hyp = NERHypothesis(mentions=[], source="direct")
        if direct_seed_enabled:
            started = time.perf_counter()
            self._agent_progress("ner_direct_agent", "start", sample_id=sample_id)
            direct_seed_hyp, cost, trace = self.ner_agent.run_direct(text, self.schema)
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            _accumulate_cost(total_cost, cost)
            traces["ner_direct"] = trace
            self._agent_progress(
                "ner_direct_agent",
                "done",
                sample_id=sample_id,
                llm_calls=cost.calls,
                mentions=len(direct_seed_hyp.mentions),
                elapsed_ms=elapsed_ms,
            )
        else:
            traces["ner_direct"] = {"disabled": True}

        started = time.perf_counter()
        self._agent_progress("candidate_agent", "start", sample_id=sample_id)
        candidate_set, cost, trace = self.candidate_agent.run(
            text,
            self.schema,
            seed_mentions=direct_seed_hyp.mentions,
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        _accumulate_cost(total_cost, cost)
        traces["candidate"] = trace
        self._agent_progress(
            "candidate_agent",
            "done",
            sample_id=sample_id,
            spans=len(candidate_set.spans),
            llm_calls=cost.calls,
            elapsed_ms=elapsed_ms,
        )
        communications.append(
            {
                "from": "candidate_agent",
                "to": "pipeline",
                "message_type": "candidate_set",
                "span_count": len(candidate_set.spans),
            }
        )
        if direct_seed_enabled and direct_seed_hyp.mentions:
            added_from_seed = _merge_candidate_set_with_mentions(candidate_set, direct_seed_hyp.mentions)
            traces["direct_seed_candidate_merge"] = {
                "enabled": True,
                "seed_mentions": len(direct_seed_hyp.mentions),
                "added_span_ids": added_from_seed,
                "added_count": len(added_from_seed),
            }
            communications.append(
                {
                    "from": "ner_direct_agent",
                    "to": "candidate_agent",
                    "message_type": "seed_mentions",
                    "mention_count": len(direct_seed_hyp.mentions),
                    "added_span_count": len(added_from_seed),
                }
            )
        else:
            traces["direct_seed_candidate_merge"] = {
                "enabled": direct_seed_enabled,
                "seed_mentions": len(direct_seed_hyp.mentions),
                "added_span_ids": [],
                "added_count": 0,
            }

        normalize_candidate_boundaries = bool(
            self.pipeline_cfg.get("normalize_candidate_boundaries", False)
        )
        boundary_left_modifiers = self.pipeline_cfg.get("boundary_left_modifiers", [])
        boundary_right_modifiers = self.pipeline_cfg.get("boundary_right_modifiers", [])
        boundary_enable_hyphen_left = bool(
            self.pipeline_cfg.get("boundary_enable_hyphen_left", True)
        )
        boundary_max_right_expansion_steps = int(
            self.pipeline_cfg.get("boundary_max_right_expansion_steps", 1)
        )
        if normalize_candidate_boundaries:
            changed, norm_trace = _normalize_candidate_span_boundaries(
                candidate_set=candidate_set,
                text=text,
                left_modifiers=[str(x) for x in boundary_left_modifiers],
                right_modifiers=[str(x) for x in boundary_right_modifiers],
                enable_hyphen_left=boundary_enable_hyphen_left,
                max_right_expansion_steps=boundary_max_right_expansion_steps,
            )
            traces["candidate_boundary_normalization"] = norm_trace
            if changed:
                communications.append(
                    {
                        "from": "pipeline",
                        "to": "all_agents",
                        "message_type": "candidate_span_boundary_update",
                        "updated_spans": changed,
                    }
                )
        else:
            traces["candidate_boundary_normalization"] = {
                "enabled": False,
                "updated_span_ids": [],
                "updated_count": 0,
            }

        memory_items: list[dict[str, Any]] = []
        if self.memory_store is not None:
            memory_items = self.memory_store.retrieve(
                query=text,
                top_k=int(self.memory_cfg.get("top_k", 3)),
                include_candidate=bool(
                    self.memory_cfg.get("retrieval_include_candidate", True)
                ),
                min_seen_count=int(self.memory_cfg.get("retrieval_min_seen_count", 1)),
                min_confidence=float(self.memory_cfg.get("retrieval_min_confidence", 0.0)),
            )
        traces["memory_retrieve"] = memory_items

        use_expert = not bool(self.ablations.get("w_o_expert", False))
        use_re = True
        use_rag = not bool(self.ablations.get("w_o_rag", False))
        parallelize_independent_branches = bool(
            self.pipeline_cfg.get("parallelize_independent_branches", True)
        )
        allow_expert_span_aug = bool(
            self.pipeline_cfg.get("allow_expert_span_augmentation", False)
        )
        allow_re_span_aug = bool(self.pipeline_cfg.get("allow_re_span_augmentation", False))
        rerun_after_aug = bool(self.pipeline_cfg.get("rerun_after_span_augmentation", True))
        max_aug_spans = int(self.pipeline_cfg.get("max_augmented_spans_per_sample", 20))
        augmentation_requires_seed = bool(
            self.pipeline_cfg.get("augmentation_requires_seed_candidate", False)
        )
        augmentation_min_conf = float(self.pipeline_cfg.get("augmentation_min_confidence", 0.0))
        augmentation_require_schema_hint = bool(
            self.pipeline_cfg.get("augmentation_require_schema_type_hint", False)
        )
        augmentation_allow_no_seed = bool(
            self.pipeline_cfg.get("augmentation_allow_no_seed_high_conf", False)
        )
        augmentation_no_seed_min_conf = float(
            self.pipeline_cfg.get("augmentation_no_seed_min_confidence", 0.9)
        )
        augmentation_reject_negative_rationale = bool(
            self.pipeline_cfg.get("augmentation_reject_negative_rationale", False)
        )
        augmentation_require_evidence = bool(
            self.pipeline_cfg.get("augmentation_require_evidence", False)
        )
        augmentation_require_evidence_anchor = bool(
            self.pipeline_cfg.get("augmentation_require_evidence_anchor", False)
        )
        augmentation_max_tokens = int(self.pipeline_cfg.get("augmentation_max_tokens", 0))
        augmentation_entity_like_only = bool(
            self.pipeline_cfg.get("augmentation_entity_like_only", False)
        )
        raw_no_seed_sources = self.pipeline_cfg.get(
            "augmentation_no_seed_allowed_sources", ["expert"]
        )
        if isinstance(raw_no_seed_sources, list):
            augmentation_no_seed_sources = {str(x).lower() for x in raw_no_seed_sources}
        else:
            augmentation_no_seed_sources = {"expert"}

        relation_schema_present = bool(self.schema.relation_constraints)
        y_exp = NERHypothesis(mentions=[], source="expert")
        y_re = NERHypothesis(
            mentions=[],
            source="re" if relation_schema_present else "in_context",
        )
        expert_retrieval_plan: dict[str, Any] = {}
        rag_handoff: dict[str, Any] = {}
        constraints = ExpertConstraints()
        relations: list[Relation] = []
        re_structure_support = ExpertConstraints()
        traces["branch_parallelism"] = {
            "enabled": parallelize_independent_branches,
            "expert_branch_parallel": False,
            "secondary_branch_parallel": False,
            "ner_branch_parallel": False,
        }

        if parallelize_independent_branches and use_expert and use_re:
            with ThreadPoolExecutor(max_workers=2) as executor:
                expert_future = executor.submit(
                    self._run_expert_branch,
                    sample_id=sample_id,
                    text=text,
                    candidate_set=candidate_set,
                    memory_items=memory_items,
                    use_expert=use_expert,
                    use_rag=use_rag,
                    allow_span_proposals=allow_expert_span_aug,
                )
                re_future = executor.submit(
                    self._run_re_branch,
                    sample_id=sample_id,
                    text=text,
                    candidate_set=candidate_set,
                    memory_items=memory_items,
                    relation_schema_present=relation_schema_present,
                    allow_span_proposals=allow_re_span_aug,
                )
                expert_result = expert_future.result()
                re_result = re_future.result()
            traces["branch_parallelism"]["expert_branch_parallel"] = True
            traces["branch_parallelism"]["secondary_branch_parallel"] = True
        else:
            expert_result = self._run_expert_branch(
                sample_id=sample_id,
                text=text,
                candidate_set=candidate_set,
                memory_items=memory_items,
                use_expert=use_expert,
                use_rag=use_rag,
                allow_span_proposals=allow_expert_span_aug,
            )
            re_result = self._run_re_branch(
                sample_id=sample_id,
                text=text,
                candidate_set=candidate_set,
                memory_items=memory_items,
                relation_schema_present=relation_schema_present,
                allow_span_proposals=allow_re_span_aug,
            )

        _accumulate_cost(total_cost, expert_result["cost"])
        _accumulate_cost(total_cost, re_result["cost"])
        traces.update(expert_result["traces"])
        traces.update(re_result["traces"])
        communications.extend(expert_result["communications"])
        communications.extend(re_result["communications"])
        expert_retrieval_plan = expert_result["expert_retrieval_plan"]
        rag_handoff = expert_result["rag_handoff"]
        constraints = expert_result["constraints"]
        relations = re_result["relations"]
        re_structure_support = re_result["re_structure_support"]

        if (
            use_re
            and not relation_schema_present
            and bool(self.pipeline_cfg.get("re_structure_support_enabled", True))
            and re_structure_support.per_span
        ):
            constraints, re_support_merge_trace = _merge_structure_support_constraints(
                base=constraints,
                support=re_structure_support,
                confidence_scale=float(
                    self.pipeline_cfg.get("re_structure_support_confidence_scale", 0.85)
                ),
                min_exclusion_confidence=float(
                    self.pipeline_cfg.get(
                        "re_structure_support_min_exclusion_confidence",
                        0.9,
                    )
                ),
            )
            traces["re_structure_support_merge"] = re_support_merge_trace
        else:
            traces["re_structure_support_merge"] = {
                "enabled": bool(self.pipeline_cfg.get("re_structure_support_enabled", True)),
                "applied_span_count": 0,
                "source_span_count": len(re_structure_support.per_span),
                "reason": "re_off"
                if not use_re
                else (
                    "schema_bound_mode"
                    if relation_schema_present
                    else "no_structure_support"
                ),
            }

        pre_augmentation_candidate_set = _copy_candidate_set(candidate_set)
        base_constraints_for_ner = constraints
        rerun_constraints_for_added = ExpertConstraints()
        augmentation_enabled = bool(allow_expert_span_aug or allow_re_span_aug)
        proposals: list[dict[str, Any]] = []
        if allow_expert_span_aug and use_expert:
            proposals.extend(traces["expert"].get("span_proposals", []))
        if allow_re_span_aug and use_re:
            proposals.extend(traces["re"].get("span_proposals", []))
        if proposals:
            raw_source_min_confidence = self.pipeline_cfg.get(
                "augmentation_source_min_confidence",
                {},
            )
            if isinstance(raw_source_min_confidence, dict):
                augmentation_source_min_confidence = {
                    str(k).lower(): float(v)
                    for k, v in raw_source_min_confidence.items()
                }
            else:
                augmentation_source_min_confidence = {}
            raw_reject_overlap_sources = self.pipeline_cfg.get(
                "augmentation_reject_overlap_sources",
                [],
            )
            if isinstance(raw_reject_overlap_sources, list):
                augmentation_reject_overlap_sources = {
                    str(x).lower() for x in raw_reject_overlap_sources
                }
            else:
                augmentation_reject_overlap_sources = set()
            proposals = _filter_span_proposals(
                proposals=proposals,
                valid_types=set(self.schema.entity_type_names),
                min_confidence=augmentation_min_conf,
                require_schema_type_hint=augmentation_require_schema_hint,
                reject_negative_rationale=augmentation_reject_negative_rationale,
                require_evidence=augmentation_require_evidence,
                require_evidence_anchor=augmentation_require_evidence_anchor,
                max_tokens=augmentation_max_tokens,
                entity_like_only=augmentation_entity_like_only,
                source_min_confidence=augmentation_source_min_confidence,
                reject_overlap_sources=augmentation_reject_overlap_sources,
                existing_spans=[
                    (span.start, span.end) for span in candidate_set.spans.values()
                ],
            )

        if augmentation_enabled:
            if augmentation_requires_seed and not candidate_set.spans:
                if augmentation_allow_no_seed:
                    proposals = [
                        p
                        for p in proposals
                        if float(p.get("confidence", 0.0)) >= augmentation_no_seed_min_conf
                        and str(p.get("source", "")).lower() in augmentation_no_seed_sources
                    ]
                    if not proposals:
                        added_ids = []
                        traces["span_augmentation"] = {
                            "enabled": True,
                            "skipped": True,
                            "reason": "no_seed_policy_filtered",
                            "proposals_total": 0,
                            "proposals_used": 0,
                            "added_count": 0,
                            "added_span_ids": [],
                        }
                    else:
                        added_ids, aug_trace = _augment_candidate_set(
                            candidate_set,
                            text,
                            proposals,
                            max_new_spans=max_aug_spans,
                        )
                        aug_trace["no_seed_mode"] = True
                        traces["span_augmentation"] = aug_trace
                else:
                    added_ids = []
                    traces["span_augmentation"] = {
                        "enabled": True,
                        "skipped": True,
                        "reason": "no_seed_candidate",
                        "proposals_total": len(proposals),
                        "proposals_used": 0,
                        "added_count": 0,
                        "added_span_ids": [],
                    }
            else:
                added_ids, aug_trace = _augment_candidate_set(
                    candidate_set,
                    text,
                    proposals,
                    max_new_spans=max_aug_spans,
                )
                traces["span_augmentation"] = aug_trace
            communications.append(
                {
                    "from": "pipeline",
                    "to": "all_agents",
                    "message_type": "span_augmentation_result",
                    "added_spans": len(added_ids),
                    "total_candidates": len(candidate_set.spans),
                }
            )
        else:
            added_ids = []
            traces["span_augmentation"] = {
                "enabled": False,
                "added_count": 0,
                "added_span_ids": [],
                "proposals_total": 0,
                "proposals_used": 0,
            }

        if added_ids and rerun_after_aug:
            added_id_set = set(added_ids)
            if use_rag and use_expert:
                expert_retrieval_plan, cost, trace = self.expert_agent.plan_retrieval(
                    text=text,
                    candidate_set=candidate_set,
                    schema=self.schema,
                    memory_items=memory_items,
                )
                _accumulate_cost(total_cost, cost)
                traces["expert_retrieval_rerun"] = trace
                rag_handoff, cost, trace = self.rag_agent.run(
                    text=text,
                    candidate_set=candidate_set,
                    schema=self.schema,
                    memory_items=memory_items,
                    expert_retrieval_plan=expert_retrieval_plan,
                )
                _accumulate_cost(total_cost, cost)
                traces["rag_rerun"] = trace

            if use_expert:
                base_constraint_count = len(constraints.per_span)
                rerun_constraints, cost, trace = self.expert_agent.run(
                    text=text,
                    candidate_set=candidate_set,
                    schema=self.schema,
                    memory_items=memory_items,
                    rag_handoff=rag_handoff,
                    allow_span_proposals=False,
                )
                _accumulate_cost(total_cost, cost)
                traces["expert_rerun"] = trace
                rerun_constraints_for_added = rerun_constraints
                constraints, applied_span_ids = _overlay_constraints_for_span_ids(
                    constraints,
                    rerun_constraints,
                    added_id_set,
                )
                traces["expert_rerun_merge"] = {
                    "mode": "added_span_only",
                    "base_constrained_spans": base_constraint_count,
                    "rerun_constrained_spans": len(rerun_constraints.per_span),
                    "applied_span_ids": sorted(applied_span_ids),
                    "applied_count": len(applied_span_ids),
                }

            if use_re:
                base_relation_count = len(relations)
                rerun_relations, rerun_structure_support, cost, trace = self.re_agent.run(
                    text,
                    candidate_set,
                    self.schema,
                    memory_items,
                    allow_span_proposals=False,
                )
                _accumulate_cost(total_cost, cost)
                traces["re_rerun"] = trace
                relations, applied_relation_count = _overlay_relations_for_span_ids(
                    relations,
                    rerun_relations,
                    added_id_set,
                )
                traces["re_rerun_merge"] = {
                    "mode": "added_span_only",
                    "base_relations": base_relation_count,
                    "rerun_relations": len(rerun_relations),
                    "applied_relation_count": applied_relation_count,
                }
                if (
                    not relation_schema_present
                    and bool(self.pipeline_cfg.get("re_structure_support_enabled", True))
                    and rerun_structure_support.per_span
                ):
                    rerun_structure_support_subset = _subset_constraints(
                        rerun_structure_support,
                        added_id_set,
                    )
                    constraints, re_rerun_support_merge_trace = _merge_structure_support_constraints(
                        base=constraints,
                        support=rerun_structure_support_subset,
                        confidence_scale=float(
                            self.pipeline_cfg.get(
                                "re_structure_support_confidence_scale",
                                0.85,
                            )
                        ),
                        min_exclusion_confidence=float(
                            self.pipeline_cfg.get(
                                "re_structure_support_min_exclusion_confidence",
                                0.9,
                            )
                        ),
                    )
                    traces["re_structure_support_rerun_merge"] = {
                        **re_rerun_support_merge_trace,
                        "mode": "added_span_only",
                    }
                else:
                    traces["re_structure_support_rerun_merge"] = {
                        "enabled": bool(
                            self.pipeline_cfg.get("re_structure_support_enabled", True)
                        ),
                        "applied_span_count": 0,
                        "source_span_count": len(rerun_structure_support.per_span),
                        "reason": "schema_bound_mode"
                        if relation_schema_present
                        else "no_structure_support",
                    }

        if not candidate_set.spans:
            traces["short_circuit"] = {
                "enabled": True,
                "reason": "no_candidate_spans_after_augmentation",
            }
            traces["ner_expert"] = {"skipped": True, "reason": "no_candidate_spans"}
            traces["ner_re"] = {"skipped": True, "reason": "no_candidate_spans"}
            traces["conflict"] = {"skipped": True, "reason": "no_candidate_spans"}
            traces["adjudicator"] = {"skipped": True, "reason": "no_candidate_spans"}
            traces["verifier"] = {"skipped": True, "reason": "no_candidate_spans"}
            traces["memory_writeback"] = []

            costs_out = {
                "calls": total_cost.calls,
                "prompt_tokens": total_cost.prompt_tokens,
                "completion_tokens": total_cost.completion_tokens,
                "total_tokens": total_cost.total_tokens,
                "latency_ms": total_cost.latency_ms,
                "debate_turns": total_cost.debate_turns,
                "debate_triggered": total_cost.debate_triggered,
                "debate_trigger_rate": 0.0,
            }
            return {
                "id": sample_id,
                "text": text,
                "mentions": [],
                "traces": traces,
                "costs": costs_out,
            }
        traces["short_circuit"] = {"enabled": False}

        if parallelize_independent_branches and use_expert and use_re:
            with ThreadPoolExecutor(max_workers=2) as executor:
                expert_ner_future = executor.submit(
                    self._run_expert_ner_branch,
                    sample_id=sample_id,
                    text=text,
                    use_expert=use_expert,
                    candidate_set=candidate_set,
                    constraints=constraints,
                    pre_augmentation_candidate_set=pre_augmentation_candidate_set,
                    rerun_constraints_for_added=rerun_constraints_for_added,
                    added_ids=added_ids,
                    rerun_after_aug=rerun_after_aug,
                )
                secondary_ner_future = executor.submit(
                    self._run_secondary_ner_branch,
                    sample_id=sample_id,
                    text=text,
                    candidate_set=candidate_set,
                    relation_schema_present=relation_schema_present,
                    relations=relations,
                    re_structure_support=re_structure_support,
                )
                expert_ner_result = expert_ner_future.result()
                secondary_ner_result = secondary_ner_future.result()
            traces["branch_parallelism"]["ner_branch_parallel"] = True
        else:
            expert_ner_result = self._run_expert_ner_branch(
                sample_id=sample_id,
                text=text,
                use_expert=use_expert,
                candidate_set=candidate_set,
                constraints=constraints,
                pre_augmentation_candidate_set=pre_augmentation_candidate_set,
                rerun_constraints_for_added=rerun_constraints_for_added,
                added_ids=added_ids,
                rerun_after_aug=rerun_after_aug,
            )
            secondary_ner_result = self._run_secondary_ner_branch(
                sample_id=sample_id,
                text=text,
                candidate_set=candidate_set,
                relation_schema_present=relation_schema_present,
                relations=relations,
                re_structure_support=re_structure_support,
            )

        _accumulate_cost(total_cost, expert_ner_result["cost"])
        _accumulate_cost(total_cost, secondary_ner_result["cost"])
        y_exp = expert_ner_result["hypothesis"]
        y_re = secondary_ner_result["hypothesis"]
        traces["ner_expert"] = expert_ner_result["trace"]
        traces["ner_re"] = secondary_ner_result["trace"]

        if relation_schema_present and bool(
            self.pipeline_cfg.get("re_collab_filter_enabled", False)
        ):
            y_re, re_filter_trace = _filter_re_hypothesis_for_collaboration(
                text=text,
                y_re=y_re,
                y_exp=y_exp if use_expert else None,
                min_confidence=float(
                    self.pipeline_cfg.get("re_collab_min_confidence", 0.0)
                ),
                require_evidence=bool(
                    self.pipeline_cfg.get("re_collab_require_evidence", False)
                ),
                max_re_only_additions=int(
                    self.pipeline_cfg.get("re_collab_max_re_only_additions", 9999)
                ),
                expert_override_margin=float(
                    self.pipeline_cfg.get("re_collab_expert_override_margin", 0.0)
                ),
                require_type_agreement_on_shared_span=bool(
                    self.pipeline_cfg.get(
                        "re_collab_require_type_agreement_on_shared_span",
                        False,
                    )
                ),
            )
            traces["re_collab_filter"] = re_filter_trace
        else:
            traces["re_collab_filter"] = {
                "enabled": bool(relation_schema_present)
                and bool(self.pipeline_cfg.get("re_collab_filter_enabled", False)),
                "input_count": len(y_re.mentions),
                "output_count": len(y_re.mentions),
                "dropped_low_conf": 0,
                "dropped_no_evidence": 0,
                "dropped_expert_override": 0,
                "dropped_re_only_cap": 0,
                "reason": "relation_schema_absent" if not relation_schema_present else "",
            }

        mention_reject_negative_rationale = bool(
            self.pipeline_cfg.get("mention_reject_negative_rationale", False)
        )
        if mention_reject_negative_rationale:
            y_exp_mentions, dropped_exp = _drop_mentions_with_negative_rationale(y_exp.mentions)
            y_re_mentions, dropped_re = _drop_mentions_with_negative_rationale(y_re.mentions)
            y_exp = NERHypothesis(mentions=y_exp_mentions, source="expert")
            y_re = NERHypothesis(mentions=y_re_mentions, source=y_re.source)
            traces["mention_filter"] = {
                "enabled": True,
                "dropped_expert": dropped_exp,
                "dropped_re": dropped_re,
            }
        else:
            traces["mention_filter"] = {"enabled": False, "dropped_expert": 0, "dropped_re": 0}

        final_mentions: list[Mention]
        cluster_count = 0
        re_contributes_ner = use_re and bool(y_re.mentions)
        adjudication_mentions: list[Mention] = []

        if use_expert:
            adjudication_mentions.extend(y_exp.mentions)
        if re_contributes_ner:
            adjudication_mentions.extend(y_re.mentions)

        if adjudication_mentions:
            iou_threshold = float(self.pipeline_cfg.get("iou_align_threshold", 0.5))
            if use_expert and re_contributes_ner:
                clusters, conflict_trace = build_conflict_clusters(
                    y_exp, y_re, iou_threshold=iou_threshold
                )
                scored = triage_conflicts(
                    clusters,
                    y_exp.mentions,
                    y_re.mentions,
                    l2_threshold=float(self.pipeline_cfg.get("l2_threshold", 0.4)),
                    l3_threshold=float(self.pipeline_cfg.get("l3_threshold", 0.75)),
                )
                clusters = apply_risk_levels(clusters, scored)
            else:
                clusters = []
                conflict_trace = {"mode": "semantic_review_only", "alignment_pairs": []}

            clusters = _ensure_singleton_adjudication_clusters(clusters, adjudication_mentions)
            cluster_count = len(clusters)

            traces["conflict"] = {
                "clusters": [to_dict(c) for c in clusters],
                "trace": conflict_trace,
            }

            started = time.perf_counter()
            self._agent_progress("adjudicator_agent", "start", sample_id=sample_id)
            decision, cost, adj_trace = self.adjudicator.run(
                text=text,
                clusters=clusters,
                y_exp=y_exp,
                y_re=y_re
                if re_contributes_ner
                else NERHypothesis(
                    mentions=[],
                    source="re" if relation_schema_present else "in_context",
                ),
                debate_protocol=self.debate_protocol,
                enable_debate=not bool(self.ablations.get("w_o_debate", False)),
                l3_only=bool(self.pipeline_cfg.get("debate_l3_only", True)),
                review_all_mentions=bool(
                    self.pipeline_cfg.get("adjudicator_review_all_mentions", True)
                ),
                singleton_policy=str(
                    self.pipeline_cfg.get("adjudicator_singleton_policy", "legacy")
                ),
                singleton_min_confidence=float(
                    self.pipeline_cfg.get("adjudicator_singleton_min_confidence", 0.0)
                ),
                singleton_require_entity_like=bool(
                    self.pipeline_cfg.get("adjudicator_singleton_require_entity_like", False)
                ),
            )
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            _accumulate_cost(total_cost, cost)
            traces["adjudicator"] = adj_trace
            final_mentions = _merge_mentions(decision.final_mentions)
            self._agent_progress(
                "adjudicator_agent",
                "done",
                sample_id=sample_id,
                llm_calls=cost.calls,
                mentions=len(final_mentions),
                clusters=cluster_count,
                elapsed_ms=elapsed_ms,
            )
        elif use_expert:
            final_mentions = _merge_mentions(y_exp.mentions)
            traces["conflict"] = {
                "disabled": True,
                "reason": "secondary_empty",
            }
            traces["adjudicator"] = {
                "clusters": [],
                "decision": "no_mentions_after_semantic_review",
            }
        elif use_re:
            final_mentions = _merge_mentions(y_re.mentions)
            traces["conflict"] = {
                "disabled": True,
                "reason": "expert_off",
            }
            traces["adjudicator"] = {
                "clusters": [],
                "decision": "no_mentions_after_semantic_review",
            }
        else:
            final_mentions = []
            traces["conflict"] = {"disabled": True, "reason": "both_off"}
            traces["adjudicator"] = {"disabled": True, "reason": "both_off"}

        if direct_seed_enabled and direct_seed_hyp.mentions:
            final_mentions, direct_seed_trace = _merge_with_direct_seed_mentions(
                direct_mentions=direct_seed_hyp.mentions,
                candidate_mentions=final_mentions,
                min_additional_confidence=float(
                    self.pipeline_cfg.get("direct_seed_additional_min_confidence", 0.92)
                ),
                protected_confidence=float(
                    self.pipeline_cfg.get("direct_seed_protected_confidence", 0.0)
                ),
                same_type_policy=str(
                    self.pipeline_cfg.get(
                        "direct_seed_same_type_policy",
                        "prefer_direct_boundary",
                    )
                ),
                allow_cross_type_override=bool(
                    self.pipeline_cfg.get("direct_seed_allow_cross_type_override", False)
                ),
                cross_type_override_margin=float(
                    self.pipeline_cfg.get("direct_seed_cross_type_override_margin", 0.05)
                ),
                cross_type_override_min_confidence=float(
                    self.pipeline_cfg.get(
                        "direct_seed_cross_type_override_min_confidence",
                        0.75,
                    )
                ),
                cross_type_override_require_evidence=bool(
                    self.pipeline_cfg.get(
                        "direct_seed_cross_type_override_require_evidence",
                        True,
                    )
                ),
            )
        else:
            direct_seed_trace = {
                "enabled": direct_seed_enabled,
                "kept_direct": 0,
                "merged_same_type": 0,
                "exact_match_merged": 0,
                "preserved_direct_boundary": 0,
                "cross_type_overridden": 0,
                "dropped_conflicting": 0,
                "added_non_overlapping": 0,
                "skipped_low_confidence": 0,
                "same_type_policy": "disabled",
                "allow_cross_type_override": False,
            }
        traces["direct_seed_guard"] = direct_seed_trace

        rescue_enabled = bool(
            self.pipeline_cfg.get("rescue_symbolic_parts_from_hypotheses", False)
        )
        if rescue_enabled:
            final_mentions, rescue_trace = _rescue_symbolic_parts_from_hypotheses(
                text=text,
                final_mentions=final_mentions,
                source_mentions=y_exp.mentions + y_re.mentions,
                min_confidence=float(
                    self.pipeline_cfg.get("rescue_symbolic_parts_min_confidence", 0.85)
                ),
            )
            traces["symbolic_part_rescue"] = rescue_trace
        else:
            traces["symbolic_part_rescue"] = {"enabled": False, "added_count": 0}

        superspan_enabled = bool(
            self.pipeline_cfg.get("enable_candidate_superspan_promotion", False)
        )
        if superspan_enabled:
            final_mentions, superspan_trace = _promote_mentions_to_candidate_superspans(
                text=text,
                mentions=final_mentions,
                candidate_set=candidate_set,
                schema=self.schema,
                max_expand_chars=int(
                    self.pipeline_cfg.get("candidate_superspan_max_expand_chars", 40)
                ),
            )
        else:
            superspan_trace = {"enabled": False, "promoted_count": 0, "promoted_span_ids": []}
        traces["candidate_superspan_promotion"] = superspan_trace

        raw_descriptor_terms = self.pipeline_cfg.get("descriptor_terms", [])
        descriptor_terms = _normalize_terms(raw_descriptor_terms)
        if bool(self.pipeline_cfg.get("descriptor_terms_from_schema", True)):
            descriptor_terms = sorted(set(descriptor_terms).union(_schema_descriptor_terms(self.schema)))

        final_mentions, postprocess_trace = _postprocess_final_mentions(
            text=text,
            mentions=final_mentions,
            enable_descriptor_expansion=bool(
                self.pipeline_cfg.get("enable_descriptor_expansion", False)
            ),
            descriptor_terms=descriptor_terms,
            descriptor_left_modifiers=[
                str(x)
                for x in (self.pipeline_cfg.get("descriptor_left_modifiers", []) or [])
                if str(x).strip()
            ],
        )
        alias_enabled = bool(
            self.pipeline_cfg.get("enable_parenthetical_alias_extraction", False)
        )
        if alias_enabled:
            final_mentions, alias_trace = _extract_parenthetical_alias_mentions(text, final_mentions)
        else:
            alias_trace = {"enabled": False, "added_count": 0, "added_span_ids": []}
        traces["final_postprocess"] = postprocess_trace
        traces["alias_extraction"] = alias_trace

        recall_bridge_enabled = bool(
            self.pipeline_cfg.get("enable_candidate_recall_bridge", False)
        )
        if recall_bridge_enabled:
            final_mentions, recall_bridge_trace = _inject_candidate_recall_mentions(
                text=text,
                candidate_set=candidate_set,
                mentions=final_mentions,
                constraints=constraints,
                schema=self.schema,
                max_added=int(self.pipeline_cfg.get("candidate_recall_max_added", 10)),
                min_confidence=float(
                    self.pipeline_cfg.get("candidate_recall_confidence", 0.78)
                ),
                skip_excluded_confidence=float(
                    self.pipeline_cfg.get(
                        "candidate_recall_skip_excluded_confidence",
                        0.9,
                    )
                ),
            )
        else:
            recall_bridge_trace = {"enabled": False, "added_count": 0, "added_span_ids": []}
        traces["candidate_recall_bridge"] = recall_bridge_trace

        lexical_injection_enabled = bool(self.pipeline_cfg.get("inject_high_value_candidates", False))
        lexical_rules = self.pipeline_cfg.get("lexical_injection_rules", []) or []
        default_ent_type = self.schema.entity_type_names[0] if self.schema.entity_type_names else "ENTITY"
        if lexical_injection_enabled and lexical_rules:
            final_mentions, injection_trace = _inject_configured_candidate_mentions(
                text=text,
                candidate_set=candidate_set,
                mentions=final_mentions,
                rules=lexical_rules,
                default_ent_type=default_ent_type,
            )
            traces["high_value_injection"] = injection_trace
        else:
            traces["high_value_injection"] = {"enabled": False, "added": 0}

        blocklist_terms = _normalize_terms(self.pipeline_cfg.get("mention_blocklist", []))
        final_mentions, blocklist_trace = _drop_blocklisted_mentions(
            final_mentions,
            blocked_terms=set(blocklist_terms),
        )
        traces["blocklist_filter"] = blocklist_trace

        low_info_filter_enabled = bool(
            self.pipeline_cfg.get("enable_low_information_filter", False)
        )
        if low_info_filter_enabled:
            final_mentions, low_info_trace = _drop_low_information_mentions(
                final_mentions,
                titlecase_confidence_threshold=float(
                    self.pipeline_cfg.get(
                        "low_information_titlecase_confidence_threshold",
                        0.9,
                    )
                ),
                descriptor_confidence_threshold=float(
                    self.pipeline_cfg.get(
                        "low_information_descriptor_confidence_threshold",
                        0.9,
                    )
                ),
            )
        else:
            low_info_trace = {"enabled": False, "dropped": 0, "dropped_span_ids": []}
        traces["low_information_filter"] = low_info_trace

        traces["policy_removed_modules"] = [
            "lexical_pattern_modules",
            "boundary_override_modules",
            "post_repair_modules",
        ]

        final_refine_enabled = bool(self.pipeline_cfg.get("enable_final_refine_pass", True))
        if final_refine_enabled:
            final_mentions, final_refine_trace = _postprocess_final_mentions(
                text=text,
                mentions=final_mentions,
                enable_descriptor_expansion=False,
                descriptor_terms=[],
                descriptor_left_modifiers=[],
            )
        else:
            final_refine_trace = {"enabled": False, "final_count": len(final_mentions)}
        traces["final_refine_pass"] = final_refine_trace

        boundary_align_enabled = bool(
            self.pipeline_cfg.get("enable_candidate_boundary_alignment", False)
        )
        if boundary_align_enabled:
            final_mentions, boundary_align_trace = _align_mentions_to_candidate_boundaries(
                text=text,
                mentions=final_mentions,
                candidate_set=candidate_set,
                schema=self.schema,
                max_expand_chars=int(
                    self.pipeline_cfg.get("candidate_boundary_alignment_max_expand_chars", 64)
                ),
                max_tokens=int(
                    self.pipeline_cfg.get("candidate_boundary_alignment_max_tokens", 14)
                ),
                min_score_gain=float(
                    self.pipeline_cfg.get("candidate_boundary_alignment_min_score_gain", 0.08)
                ),
            )
        else:
            boundary_align_trace = {"enabled": False, "aligned_count": 0, "aligned_span_ids": []}
        traces["candidate_boundary_alignment"] = boundary_align_trace

        expert_restore_enabled = bool(
            self.pipeline_cfg.get("enable_expert_support_restore", False)
        )
        if expert_restore_enabled and use_expert:
            final_mentions, expert_restore_trace = _restore_high_support_expert_mentions(
                text=text,
                current_mentions=final_mentions,
                expert_mentions=y_exp.mentions,
                constraints=constraints,
                schema=self.schema,
                max_added=int(self.pipeline_cfg.get("expert_support_restore_max_added", 12)),
                min_mention_confidence=float(
                    self.pipeline_cfg.get("expert_support_restore_min_mention_confidence", 0.82)
                ),
                min_constraint_confidence=float(
                    self.pipeline_cfg.get("expert_support_restore_min_constraint_confidence", 0.6)
                ),
            )
        else:
            expert_restore_trace = {"enabled": False, "added_count": 0, "added_span_ids": []}
        traces["expert_support_restore"] = expert_restore_trace

        expert_boundary_calibration_enabled = bool(
            self.pipeline_cfg.get("enable_expert_evidence_boundary_calibration", False)
        )
        if expert_boundary_calibration_enabled and use_expert:
            final_mentions, expert_boundary_trace = _calibrate_mentions_with_expert_evidence(
                text=text,
                mentions=final_mentions,
                constraints=constraints,
                schema=self.schema,
                min_score_gain=float(
                    self.pipeline_cfg.get(
                        "expert_evidence_boundary_calibration_min_score_gain",
                        0.12,
                    )
                ),
            )
        else:
            expert_boundary_trace = {"enabled": False, "calibrated_count": 0, "calibrated_span_ids": []}
        traces["expert_evidence_boundary_calibration"] = expert_boundary_trace

        reapply_direct_seed = bool(
            self.pipeline_cfg.get("reapply_direct_seed_after_postprocess", True)
        )
        if reapply_direct_seed and direct_seed_enabled and direct_seed_hyp.mentions:
            final_mentions, direct_seed_post_trace = _merge_with_direct_seed_mentions(
                direct_mentions=direct_seed_hyp.mentions,
                candidate_mentions=final_mentions,
                min_additional_confidence=float(
                    self.pipeline_cfg.get("direct_seed_additional_min_confidence", 0.92)
                ),
                protected_confidence=float(
                    self.pipeline_cfg.get("direct_seed_protected_confidence", 0.0)
                ),
                same_type_policy=str(
                    self.pipeline_cfg.get(
                        "direct_seed_same_type_policy",
                        "prefer_direct_boundary",
                    )
                ),
                allow_cross_type_override=bool(
                    self.pipeline_cfg.get("direct_seed_allow_cross_type_override", False)
                ),
                cross_type_override_margin=float(
                    self.pipeline_cfg.get("direct_seed_cross_type_override_margin", 0.05)
                ),
                cross_type_override_min_confidence=float(
                    self.pipeline_cfg.get(
                        "direct_seed_cross_type_override_min_confidence",
                        0.75,
                    )
                ),
                cross_type_override_require_evidence=bool(
                    self.pipeline_cfg.get(
                        "direct_seed_cross_type_override_require_evidence",
                        True,
                    )
                ),
            )
        else:
            direct_seed_post_trace = {
                "enabled": reapply_direct_seed and direct_seed_enabled,
                "kept_direct": 0,
                "merged_same_type": 0,
                "exact_match_merged": 0,
                "preserved_direct_boundary": 0,
                "cross_type_overridden": 0,
                "dropped_conflicting": 0,
                "added_non_overlapping": 0,
                "skipped_low_confidence": 0,
                "same_type_policy": "disabled",
                "allow_cross_type_override": False,
            }
        traces["direct_seed_guard_postprocess"] = direct_seed_post_trace

        disambiguation_enabled = bool(
            self.pipeline_cfg.get("enable_disambiguation_agent", False)
        )
        if disambiguation_enabled:
            lock_conf = float(self.pipeline_cfg.get("disambiguation_lock_confidence", 0.92))
            lock_non_pipeline = bool(
                self.pipeline_cfg.get("disambiguation_lock_if_non_pipeline", True)
            )
            lock_direct_anchor = bool(
                self.pipeline_cfg.get("disambiguation_lock_if_direct_anchor", True)
            )
            protect_direct_anchor_from_drop = bool(
                self.pipeline_cfg.get("disambiguation_protect_direct_anchor_from_drop", True)
            )
            direct_anchor_iou = float(
                self.pipeline_cfg.get("disambiguation_direct_anchor_iou", 0.8)
            )
            direct_anchor_min_conf = float(
                self.pipeline_cfg.get(
                    "disambiguation_direct_anchor_min_confidence",
                    0.9,
                )
            )
            pipeline_tags = (
                "pipeline:",
                "candidate_recall_bridge",
            )
            locked_mentions: list[Mention] = []
            protected_review_mentions: list[Mention] = []
            review_mentions: list[Mention] = []
            for m in final_mentions:
                rationale_l = (m.rationale or "").lower()
                from_pipeline = any(tag in rationale_l for tag in pipeline_tags)
                anchored_by_direct = False
                if lock_direct_anchor and direct_seed_hyp.mentions:
                    anchored_by_direct = any(
                        direct.ent_type == m.ent_type
                        and direct.confidence >= direct_anchor_min_conf
                        and span_iou(direct.span, m.span) >= direct_anchor_iou
                        for direct in direct_seed_hyp.mentions
                    )
                if anchored_by_direct:
                    locked_mentions.append(m)
                    continue
                if m.confidence >= lock_conf and (not lock_non_pipeline or not from_pipeline):
                    locked_mentions.append(m)
                else:
                    anchored_for_protection = False
                    if direct_seed_hyp.mentions:
                        anchored_for_protection = any(
                            direct.ent_type == m.ent_type
                            and direct.confidence >= direct_anchor_min_conf
                            and span_iou(direct.span, m.span) >= direct_anchor_iou
                            for direct in direct_seed_hyp.mentions
                        )
                    if anchored_for_protection and protect_direct_anchor_from_drop:
                        protected_review_mentions.append(m)
                    else:
                        review_mentions.append(m)

            started = time.perf_counter()
            self._agent_progress("disambiguation_agent", "start", sample_id=sample_id)
            protected_mentions, protected_trace, protected_cost = self.disambiguation_agent.run(
                text=text,
                mentions=protected_review_mentions,
                schema=self.schema,
                allow_drop=False,
            )
            reviewed_mentions, disamb_trace, disamb_cost = self.disambiguation_agent.run(
                text=text,
                mentions=review_mentions,
                schema=self.schema,
                allow_drop=bool(self.pipeline_cfg.get("disambiguation_allow_drop", False)),
            )
            final_mentions = locked_mentions + protected_mentions + reviewed_mentions
            _accumulate_cost(total_cost, protected_cost)
            _accumulate_cost(total_cost, disamb_cost)
            disamb_trace["locked_count"] = len(locked_mentions)
            disamb_trace["protected_review_count"] = len(protected_review_mentions)
            disamb_trace["review_count"] = len(review_mentions) + len(protected_review_mentions)
            disamb_trace["protected_review_output_count"] = len(protected_mentions)
            disamb_trace["protected_review_dropped"] = int(protected_trace.get("dropped", 0) or 0)
            disamb_trace["protected_review_adjusted"] = int(protected_trace.get("adjusted", 0) or 0)
            traces["disambiguation_agent"] = disamb_trace
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            self._agent_progress(
                "disambiguation_agent",
                "done",
                sample_id=sample_id,
                llm_calls=protected_cost.calls + disamb_cost.calls,
                mentions=len(final_mentions),
                elapsed_ms=elapsed_ms,
            )
            communications.append(
                {
                    "from": "disambiguation_agent",
                    "to": "pipeline",
                    "message_type": "disambiguation_update",
                    "output_count": len(final_mentions),
                }
            )
        else:
            traces["disambiguation_agent"] = {"enabled": False, "output_count": len(final_mentions)}

        final_min_confidence = float(self.pipeline_cfg.get("final_min_confidence", 0.0))
        if final_min_confidence > 0.0:
            before = len(final_mentions)
            final_mentions = [m for m in final_mentions if m.confidence >= final_min_confidence]
            traces["final_confidence_filter"] = {
                "enabled": True,
                "threshold": final_min_confidence,
                "dropped": before - len(final_mentions),
            }
        else:
            traces["final_confidence_filter"] = {
                "enabled": False,
                "threshold": 0.0,
                "dropped": 0,
            }

        symbol_canon_enabled = bool(
            self.pipeline_cfg.get("enable_symbol_boundary_canonicalization", False)
        )
        if symbol_canon_enabled:
            final_mentions, symbol_canon_trace = _canonicalize_symbolic_boundaries(
                text=text,
                mentions=final_mentions,
            )
        else:
            symbol_canon_trace = {
                "enabled": False,
                "adjusted_count": 0,
                "adjusted_span_ids": [],
            }
        traces["symbol_boundary_canonicalization"] = symbol_canon_trace

        if not bool(self.ablations.get("w_o_verifier", False)):
            started = time.perf_counter()
            self._agent_progress("verifier_agent", "start", sample_id=sample_id)
            verified_mentions, report, cost = self.verifier.verify_mentions(
                text, final_mentions, self.schema
            )
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            _accumulate_cost(total_cost, cost)
            traces["verifier"] = report
            final_mentions = verified_mentions
            verifier_pass_for_memory = True
            self._agent_progress(
                "verifier_agent",
                "done",
                sample_id=sample_id,
                llm_calls=cost.calls,
                mentions=len(final_mentions),
                elapsed_ms=elapsed_ms,
            )
        else:
            traces["verifier"] = {"disabled": True}
            verifier_pass_for_memory = False

        memory_writes = []
        if self.memory_store is not None:
            for m in final_mentions:
                result = self.memory_store.writeback(
                    kind="term",
                    key=m.span.text,
                    value={
                        "ent_type": m.ent_type,
                        "span_id": m.span_id,
                        "rationale": m.rationale,
                    },
                    confidence=m.confidence,
                    verifier_pass=verifier_pass_for_memory,
                    promote_threshold=int(self.memory_cfg.get("promote_threshold", 2)),
                    min_confidence=float(self.memory_cfg.get("min_confidence", 0.7)),
                )
                memory_writes.append({"span_id": m.span_id, "result": result})
        traces["memory_writeback"] = memory_writes

        trigger_rate = (total_cost.debate_triggered / cluster_count) if cluster_count else 0.0
        costs_out = {
            "calls": total_cost.calls,
            "prompt_tokens": total_cost.prompt_tokens,
            "completion_tokens": total_cost.completion_tokens,
            "total_tokens": total_cost.total_tokens,
            "latency_ms": total_cost.latency_ms,
            "debate_turns": total_cost.debate_turns,
            "debate_triggered": total_cost.debate_triggered,
            "debate_trigger_rate": trigger_rate,
        }

        return {
            "id": sample_id,
            "text": text,
            "mentions": [to_dict(m) for m in final_mentions],
            "traces": traces,
            "costs": costs_out,
        }


def _is_retryable_pipeline_exception(exc: Exception) -> bool:
    message = str(exc).lower()
    retry_markers = [
        "network error",
        "transport error",
        "timed out",
        "timeout",
        "connection reset",
        "winerror 10054",
        "http error 429",
        "http error 500",
        "http error 502",
        "http error 503",
        "http error 504",
        "json parsing failed",
    ]
    return any(marker in message for marker in retry_markers)


def _filter_re_hypothesis_for_collaboration(
    text: str,
    y_re: NERHypothesis,
    y_exp: NERHypothesis | None,
    min_confidence: float = 0.0,
    require_evidence: bool = False,
    max_re_only_additions: int = 9999,
    expert_override_margin: float = 0.0,
    require_type_agreement_on_shared_span: bool = False,
) -> tuple[NERHypothesis, dict[str, Any]]:
    input_mentions = list(y_re.mentions)
    if not input_mentions:
        return y_re, {
            "enabled": True,
            "input_count": 0,
            "output_count": 0,
            "dropped_low_conf": 0,
            "dropped_no_evidence": 0,
            "dropped_expert_override": 0,
            "dropped_re_only_cap": 0,
        }

    expert_by_sid = {m.span_id: m for m in (y_exp.mentions if y_exp is not None else [])}
    kept: list[Mention] = []
    re_only_pool: list[Mention] = []
    dropped_low_conf = 0
    dropped_no_evidence = 0
    dropped_expert_override = 0
    dropped_re_only_cap = 0

    for m in input_mentions:
        if m.confidence < min_confidence:
            dropped_low_conf += 1
            continue
        if require_evidence and not m.evidence:
            dropped_no_evidence += 1
            continue

        exp = expert_by_sid.get(m.span_id)
        if exp is None:
            re_only_pool.append(m)
            continue

        if require_type_agreement_on_shared_span and m.ent_type != exp.ent_type:
            dropped_expert_override += 1
            continue

        # Prefer expert typing when confidence is close and span_id is shared.
        if m.ent_type != exp.ent_type and (exp.confidence + expert_override_margin) >= m.confidence:
            dropped_expert_override += 1
            continue
        kept.append(m)

    if max_re_only_additions < 0:
        max_re_only_additions = 0
    if re_only_pool:
        ranked = sorted(
            re_only_pool,
            key=lambda x: (
                x.confidence,
                -(x.span.end - x.span.start),
            ),
            reverse=True,
        )
        selected = ranked[:max_re_only_additions]
        dropped_re_only_cap = max(0, len(ranked) - len(selected))
        kept.extend(selected)

    kept = sorted(
        kept,
        key=lambda x: (
            x.span.start,
            x.span.end,
            -x.confidence,
            x.span_id,
        ),
    )
    hyp = NERHypothesis(mentions=kept, source=y_re.source)
    trace = {
        "enabled": True,
        "input_count": len(input_mentions),
        "output_count": len(kept),
        "dropped_low_conf": dropped_low_conf,
        "dropped_no_evidence": dropped_no_evidence,
        "dropped_expert_override": dropped_expert_override,
        "dropped_re_only_cap": dropped_re_only_cap,
    }
    return hyp, trace


def write_predictions(records: list[dict[str, Any]], pred_path: str | Path) -> None:
    path = Path(pred_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in records:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _merge_mentions(mentions: list[Mention]) -> list[Mention]:
    # First-pass merge by span_id, keeping the highest-confidence hypothesis for that id.
    best: dict[str, Mention] = {}
    for m in mentions:
        prev = best.get(m.span_id)
        if prev is None or m.confidence > prev.confidence:
            best[m.span_id] = m
    return list(best.values())


def _merge_candidate_set_with_mentions(
    candidate_set: CandidateSet,
    mentions: list[Mention],
) -> list[str]:
    by_offsets = {(span.start, span.end): sid for sid, span in candidate_set.spans.items()}
    next_idx = _next_aug_index(candidate_set.spans.keys())
    added_ids: list[str] = []
    for mention in mentions:
        key = (mention.span.start, mention.span.end)
        if key in by_offsets:
            continue
        span_id = f"sp_seed_{next_idx:04d}"
        while span_id in candidate_set.spans:
            next_idx += 1
            span_id = f"sp_seed_{next_idx:04d}"
        candidate_set.spans[span_id] = Span(
            text=mention.span.text,
            start=mention.span.start,
            end=mention.span.end,
            provenance={"source": "direct_seed", "mention_span_id": mention.span_id},
        )
        by_offsets[key] = span_id
        added_ids.append(span_id)
        next_idx += 1
    candidate_set.has_entity = bool(candidate_set.spans)
    return added_ids


def _copy_candidate_set(candidate_set: CandidateSet) -> CandidateSet:
    return CandidateSet(
        has_entity=candidate_set.has_entity,
        spans=dict(candidate_set.spans),
    )


def _subset_candidate_set(candidate_set: CandidateSet, span_ids: set[str]) -> CandidateSet:
    spans = {sid: span for sid, span in candidate_set.spans.items() if sid in span_ids}
    return CandidateSet(has_entity=bool(spans), spans=spans)


def _overlay_constraints_for_span_ids(
    base: ExpertConstraints,
    overlay: ExpertConstraints,
    span_ids: set[str],
) -> tuple[ExpertConstraints, list[str]]:
    if not span_ids:
        return base, []
    merged = ExpertConstraints(
        terminology=list(base.terminology),
        triggers=list(base.triggers),
        per_span=dict(base.per_span),
    )
    applied: list[str] = []
    for span_id in span_ids:
        if span_id not in overlay.per_span:
            continue
        merged.per_span[span_id] = overlay.per_span[span_id]
        applied.append(span_id)
    return merged, applied


def _subset_constraints(constraints: ExpertConstraints, span_ids: set[str]) -> ExpertConstraints:
    return ExpertConstraints(
        terminology=list(constraints.terminology),
        triggers=list(constraints.triggers),
        per_span={sid: val for sid, val in constraints.per_span.items() if sid in span_ids},
    )


def _merge_structure_support_constraints(
    base: ExpertConstraints,
    support: ExpertConstraints,
    confidence_scale: float = 0.85,
    min_exclusion_confidence: float = 0.9,
) -> tuple[ExpertConstraints, dict[str, Any]]:
    merged = ExpertConstraints(
        terminology=list(base.terminology),
        triggers=list(base.triggers),
        per_span=dict(base.per_span),
    )
    applied_span_count = 0
    added_new_spans = 0
    enriched_existing_spans = 0

    for span_id, support_constraint in support.per_span.items():
        scaled_confidence = max(0.0, min(1.0, float(support_constraint.confidence) * float(confidence_scale)))
        if span_id not in merged.per_span:
            merged.per_span[span_id] = SpanConstraint(
                candidate_types=list(support_constraint.candidate_types),
                excluded_types=list(support_constraint.excluded_types)
                if scaled_confidence >= float(min_exclusion_confidence)
                else [],
                boundary_ops=[],
                evidence=list(support_constraint.evidence),
                confidence=scaled_confidence,
                rationale=str(support_constraint.rationale),
            )
            added_new_spans += 1
            applied_span_count += 1
            continue

        base_constraint = merged.per_span[span_id]
        candidate_types = list(base_constraint.candidate_types)
        if not candidate_types:
            candidate_types = list(support_constraint.candidate_types)

        excluded_types = list(base_constraint.excluded_types)
        if scaled_confidence >= float(min_exclusion_confidence):
            for item in support_constraint.excluded_types:
                if item not in excluded_types:
                    excluded_types.append(item)

        evidence = list(base_constraint.evidence)
        seen_evidence = {(ev.quote, ev.start, ev.end) for ev in evidence}
        for ev in support_constraint.evidence:
            key = (ev.quote, ev.start, ev.end)
            if key in seen_evidence:
                continue
            evidence.append(ev)
            seen_evidence.add(key)

        rationale_parts = [part for part in [base_constraint.rationale, support_constraint.rationale] if part]
        merged.per_span[span_id] = SpanConstraint(
            candidate_types=candidate_types,
            excluded_types=excluded_types,
            boundary_ops=list(base_constraint.boundary_ops),
            evidence=evidence,
            confidence=max(float(base_constraint.confidence), scaled_confidence),
            rationale=" | ".join(rationale_parts),
        )
        enriched_existing_spans += 1
        applied_span_count += 1

    return merged, {
        "enabled": True,
        "source_span_count": len(support.per_span),
        "applied_span_count": applied_span_count,
        "added_new_spans": added_new_spans,
        "enriched_existing_spans": enriched_existing_spans,
        "confidence_scale": float(confidence_scale),
        "min_exclusion_confidence": float(min_exclusion_confidence),
    }


def _ensure_singleton_adjudication_clusters(
    clusters: list[ConflictCluster],
    mentions: list[Mention],
) -> list[ConflictCluster]:
    merged = list(clusters)
    covered_span_ids = {span_id for cluster in merged for span_id in cluster.span_ids}
    next_idx = len(merged) + 1
    for mention in mentions:
        if mention.span_id in covered_span_ids:
            continue
        merged.append(
            ConflictCluster(
                cluster_id=f"cluster_{next_idx:04d}",
                span_ids=[mention.span_id],
                conflicts=["semantic_review"],
                risk_level="L1",
                score=0.0,
            )
        )
        covered_span_ids.add(mention.span_id)
        next_idx += 1
    return merged


def _overlay_relations_for_span_ids(
    base: list[Relation],
    overlay: list[Relation],
    span_ids: set[str],
) -> tuple[list[Relation], int]:
    if not span_ids:
        return base, 0
    merged: dict[tuple[str, str, str], Relation] = {
        (rel.head_span_id, rel.rel_type, rel.tail_span_id): rel for rel in base
    }
    applied = 0
    for rel in overlay:
        if rel.head_span_id not in span_ids and rel.tail_span_id not in span_ids:
            continue
        merged[(rel.head_span_id, rel.rel_type, rel.tail_span_id)] = rel
        applied += 1
    return list(merged.values()), applied


def _merge_with_direct_seed_mentions(
    direct_mentions: list[Mention],
    candidate_mentions: list[Mention],
    min_additional_confidence: float,
    protected_confidence: float = 0.0,
    same_type_policy: str = "prefer_direct_boundary",
    allow_cross_type_override: bool = False,
    cross_type_override_margin: float = 0.05,
    cross_type_override_min_confidence: float = 0.75,
    cross_type_override_require_evidence: bool = True,
) -> tuple[list[Mention], dict[str, Any]]:
    final: list[Mention] = []
    used_candidate_ids: set[str] = set()
    merged_same_type = 0
    dropped_conflicting = 0
    skipped_low_confidence = 0
    added_non_overlapping = 0
    exact_match_merged = 0
    preserved_direct_boundary = 0
    cross_type_overridden = 0

    for direct in direct_mentions:
        overlaps = [
            cand
            for cand in candidate_mentions
            if span_iou(direct.span, cand.span) > 0.0
        ]
        same_type = [cand for cand in overlaps if cand.ent_type == direct.ent_type]
        if same_type:
            exact_same = [
                cand
                for cand in same_type
                if cand.span.start == direct.span.start and cand.span.end == direct.span.end
            ]
            if exact_same:
                best = max(exact_same + [direct], key=lambda m: m.confidence)
                final.append(best)
                for item in exact_same:
                    used_candidate_ids.add(item.span_id)
                merged_same_type += 1
                exact_match_merged += 1
                continue
            if same_type_policy == "prefer_direct_boundary":
                final.append(direct)
                for item in same_type:
                    used_candidate_ids.add(item.span_id)
                    dropped_conflicting += 1
                merged_same_type += 1
                preserved_direct_boundary += 1
                continue
        if allow_cross_type_override:
            cross_type = [cand for cand in overlaps if cand.ent_type != direct.ent_type]
            eligible_cross_type = [
                cand
                for cand in cross_type
                if cand.confidence >= cross_type_override_min_confidence
                and cand.confidence >= (direct.confidence + cross_type_override_margin)
                and (not cross_type_override_require_evidence or bool(cand.evidence))
            ]
            if direct.confidence < protected_confidence and eligible_cross_type:
                best_cross = max(
                    eligible_cross_type,
                    key=lambda m: (
                        m.confidence,
                        len(m.evidence),
                        m.span.end - m.span.start,
                    ),
                )
                final.append(best_cross)
                used_candidate_ids.add(best_cross.span_id)
                for item in overlaps:
                    if item.span_id != best_cross.span_id:
                        used_candidate_ids.add(item.span_id)
                        dropped_conflicting += 1
                cross_type_overridden += 1
                continue
        final.append(direct)
        if direct.confidence >= protected_confidence:
            for item in overlaps:
                used_candidate_ids.add(item.span_id)
                dropped_conflicting += 1

    for cand in candidate_mentions:
        if cand.span_id in used_candidate_ids:
            continue
        if any(span_iou(cand.span, direct.span) > 0.0 for direct in direct_mentions):
            if cand.confidence < min_additional_confidence:
                skipped_low_confidence += 1
                continue
            dropped_conflicting += 1
            continue
        if cand.confidence < min_additional_confidence:
            skipped_low_confidence += 1
            continue
        final.append(cand)
        added_non_overlapping += 1

    trace = {
        "enabled": True,
        "kept_direct": len(direct_mentions),
        "merged_same_type": merged_same_type,
        "exact_match_merged": exact_match_merged,
        "preserved_direct_boundary": preserved_direct_boundary,
        "cross_type_overridden": cross_type_overridden,
        "dropped_conflicting": dropped_conflicting,
        "added_non_overlapping": added_non_overlapping,
        "skipped_low_confidence": skipped_low_confidence,
        "same_type_policy": same_type_policy,
        "allow_cross_type_override": allow_cross_type_override,
    }
    return _merge_mentions(final), trace


def _postprocess_final_mentions(
    text: str,
    mentions: list[Mention],
    enable_descriptor_expansion: bool = False,
    descriptor_terms: list[str] | None = None,
    descriptor_left_modifiers: list[str] | None = None,
) -> tuple[list[Mention], dict[str, Any]]:
    if not mentions:
        return [], {
            "enabled": True,
            "input_count": 0,
            "after_span_id_merge_count": 0,
            "token_edge_normalized": 0,
            "left_noise_trimmed": 0,
            "hyphen_expanded": 0,
            "descriptor_expanded": 0,
            "offset_type_dedup_dropped": 0,
            "subsumed_dropped": 0,
            "adjacent_merged": 0,
            "coordinated_merged": 0,
            "final_count": 0,
        }

    current = _merge_mentions(mentions)
    after_span_id = len(current)
    current, token_edge_normalized = _expand_mentions_to_token_edges(text, current)
    current, expanded_hyphen = _expand_mentions_with_hyphen_suffix(text, current)
    if enable_descriptor_expansion:
        current, expanded_descriptor = _expand_mentions_with_descriptors(
            text=text,
            mentions=current,
            descriptor_terms=descriptor_terms or [],
            left_modifiers=descriptor_left_modifiers or [],
        )
    else:
        expanded_descriptor = 0
    current, left_noise_trimmed = _trim_left_noise_tokens(text, current)
    current, split_symbolic = _split_symbolic_chain_mentions(text, current)
    current, split_slash = _split_slash_coordinated_mentions(text, current)

    current, dropped_offset_type = _dedupe_mentions_by_offset_and_type(current)
    current, dropped_subsumed_a = _drop_subsumed_mentions(current)
    current, merged_adjacent = _merge_adjacent_mentions_with_connectors(text, current)
    current, merged_coordinated = _merge_coordinated_modifier_mentions(text, current)
    current, dropped_offset_type_b = _dedupe_mentions_by_offset_and_type(current)
    current, dropped_subsumed_b = _drop_subsumed_mentions(current)

    current = sorted(current, key=lambda m: (m.span.start, m.span.end, m.span_id))
    trace = {
        "enabled": True,
        "input_count": len(mentions),
        "after_span_id_merge_count": after_span_id,
        "token_edge_normalized": token_edge_normalized,
        "left_noise_trimmed": left_noise_trimmed,
        "hyphen_expanded": expanded_hyphen,
        "descriptor_expanded": expanded_descriptor,
        "symbolic_split": split_symbolic,
        "slash_split": split_slash,
        "offset_type_dedup_dropped": dropped_offset_type + dropped_offset_type_b,
        "subsumed_dropped": dropped_subsumed_a + dropped_subsumed_b,
        "adjacent_merged": merged_adjacent,
        "coordinated_merged": merged_coordinated,
        "final_count": len(current),
    }
    return current, trace


def _dedupe_mentions_by_offset_and_type(
    mentions: list[Mention],
) -> tuple[list[Mention], int]:
    best: dict[tuple[int, int, str], Mention] = {}
    for mention in mentions:
        key = (mention.span.start, mention.span.end, mention.ent_type)
        prev = best.get(key)
        if prev is None or mention.confidence > prev.confidence:
            best[key] = mention
    out = list(best.values())
    dropped = len(mentions) - len(out)
    return out, dropped


def _drop_subsumed_mentions(mentions: list[Mention]) -> tuple[list[Mention], int]:
    ordered = sorted(
        mentions,
        key=lambda m: (
            -(m.span.end - m.span.start),
            -m.confidence,
            m.span.start,
            m.span.end,
            m.span_id,
        ),
    )
    kept: list[Mention] = []
    dropped = 0

    for mention in ordered:
        should_drop = False
        for anchor in kept:
            if mention.ent_type != anchor.ent_type:
                continue
            if not _contains_span(anchor.span, mention.span):
                continue
            if _should_preserve_symbolic_submention(anchor=anchor, mention=mention):
                continue
            anchor_len = anchor.span.end - anchor.span.start
            mention_len = mention.span.end - mention.span.start
            if anchor_len <= mention_len:
                continue
            if anchor.span.start == mention.span.start or anchor.span.end == mention.span.end:
                should_drop = True
                break
            if _is_descriptor_rich(anchor.span.text) and (
                mention_len <= 8
                or (mention_len <= 14 and "-" in mention.span.text)
            ):
                should_drop = True
                break
            if anchor_len - mention_len >= 18 and mention_len <= 5:
                should_drop = True
                break
        if should_drop:
            dropped += 1
        else:
            kept.append(mention)

    return kept, dropped


def _merge_adjacent_mentions_with_connectors(
    text: str,
    mentions: list[Mention],
) -> tuple[list[Mention], int]:
    current = sorted(mentions, key=lambda m: (m.span.start, m.span.end, m.span_id))
    merged_count = 0

    while True:
        changed = False
        for i in range(len(current) - 1):
            left = current[i]
            right = current[i + 1]
            if left.ent_type != right.ent_type:
                continue
            if left.span.end > right.span.start:
                continue

            gap = text[left.span.end : right.span.start]
            if len(gap) > 5:
                continue
            connector = gap.strip()
            if connector not in {"-", ""}:
                continue
            if connector == "" and not _should_merge_adjacent_space_tokens(left.span.text, right.span.text):
                continue

            merged_start = left.span.start
            merged_end = right.span.end
            if not is_valid_offsets(text, merged_start, merged_end) or merged_start >= merged_end:
                continue
            merged_text = text[merged_start:merged_end]
            if len(merged_text) > 80:
                continue
            if connector == "-" and not _allow_hyphen_symbol_merge(left.span.text, right.span.text, merged_text):
                continue
            if not _looks_entity_like_merged_mention(merged_text):
                continue

            merged_evidence = _merge_evidence(left.evidence, right.evidence)
            merged_evidence.append(
                Evidence(quote=merged_text, start=merged_start, end=merged_end)
            )
            merged_evidence = _dedupe_evidence(merged_evidence)

            merged = Mention(
                span_id=f"{left.span_id}__merge__{right.span_id}",
                span=Span(
                    text=merged_text,
                    start=merged_start,
                    end=merged_end,
                    provenance={
                        "op": "MERGE",
                        "source_span_ids": [left.span_id, right.span_id],
                        "connector": connector,
                    },
                ),
                ent_type=left.ent_type,
                confidence=max(left.confidence, right.confidence),
                evidence=merged_evidence,
                rationale=f"pipeline:merge_adjacent:{left.span_id}+{right.span_id}",
            )

            current = current[:i] + [merged] + current[i + 2 :]
            merged_count += 1
            changed = True
            break
        if not changed:
            break

    return current, merged_count


def _canonicalize_symbolic_boundaries(
    text: str,
    mentions: list[Mention],
) -> tuple[list[Mention], dict[str, Any]]:
    if not mentions:
        return mentions, {
            "enabled": True,
            "adjusted_count": 0,
            "adjusted_span_ids": [],
        }

    out: list[Mention] = []
    adjusted_ids: list[str] = []

    for m in mentions:
        rel_tokens = [
            (mt.group(0), mt.start(), mt.end())
            for mt in re.finditer(r"[A-Za-z0-9\+\-\.]+", m.span.text)
        ]
        if len(rel_tokens) < 2:
            out.append(m)
            continue

        abs_tokens = [
            (tok, m.span.start + rs, m.span.start + re_)
            for tok, rs, re_ in rel_tokens
        ]
        token_texts = [tok for tok, _, _ in abs_tokens]
        symbol_idxs = [i for i, tok in enumerate(token_texts) if _is_compact_symbol_token(tok)]
        if not symbol_idxs:
            out.append(m)
            continue

        left = 0
        right = len(abs_tokens) - 1

        # Trim low-information lowercase wrappers before symbolic heads.
        while left < right:
            tok_l = token_texts[left]
            if not (tok_l.isalpha() and tok_l.islower() and 4 <= len(tok_l) <= 16):
                break
            next_is_wrapper_or_symbol = (
                (
                    token_texts[left + 1].isalpha()
                    and token_texts[left + 1].islower()
                    and 4 <= len(token_texts[left + 1]) <= 16
                )
                or _is_compact_symbol_token(token_texts[left + 1])
            )
            if not next_is_wrapper_or_symbol:
                break
            # Ensure a symbol remains in the kept suffix.
            if not any(idx > left and idx <= right for idx in symbol_idxs):
                break
            left += 1

        # Trim weak tails only when directly attached to a symbolic core.
        while right > left:
            tok_r = token_texts[right]
            if not (tok_r.isalpha() and tok_r.islower() and 3 <= len(tok_r) <= 12):
                break
            prev_idx = right - 1
            if prev_idx < left:
                break
            if not _is_compact_symbol_token(token_texts[prev_idx]):
                break
            right -= 1

        new_start = abs_tokens[left][1]
        new_end = abs_tokens[right][2]
        if new_start == m.span.start and new_end == m.span.end:
            out.append(m)
            continue
        if not is_valid_offsets(text, new_start, new_end) or new_start >= new_end:
            out.append(m)
            continue

        new_text = text[new_start:new_end]
        if not _is_candidate_recall_entity_like(new_text):
            out.append(m)
            continue

        updated = Mention(
            span_id=f"{m.span_id}__symcanon",
            span=Span(
                text=new_text,
                start=new_start,
                end=new_end,
                provenance={
                    "op": "SYMBOL_BOUNDARY_CANONICALIZATION",
                    "source_span_id": m.span_id,
                },
            ),
            ent_type=m.ent_type,
            confidence=m.confidence,
            evidence=_dedupe_evidence(
                _merge_evidence(
                    m.evidence,
                    [Evidence(quote=new_text, start=new_start, end=new_end)],
                )
            ),
            rationale=f"{m.rationale}|pipeline:symbol_boundary_canonicalization",
        )
        out.append(updated)
        adjusted_ids.append(updated.span_id)

    out, _ = _dedupe_mentions_by_offset_and_type(out)
    out = sorted(out, key=lambda x: (x.span.start, x.span.end, x.span_id))
    return out, {
        "enabled": True,
        "adjusted_count": len(adjusted_ids),
        "adjusted_span_ids": adjusted_ids,
    }


def _expand_mentions_to_token_edges(
    text: str,
    mentions: list[Mention],
) -> tuple[list[Mention], int]:
    out: list[Mention] = []
    changed = 0
    n = len(text)
    for m in mentions:
        s = m.span.start
        e = m.span.end
        new_s = s
        new_e = e

        # Left: if current start is inside an alnum token, extend left.
        while (
            new_s > 0
            and new_s < n
            and text[new_s].isalnum()
            and text[new_s - 1].isalnum()
        ):
            new_s -= 1

        # Right: if current end is inside an alnum token, extend right.
        while (
            new_e > 0
            and new_e < n
            and text[new_e - 1].isalnum()
            and text[new_e].isalnum()
        ):
            new_e += 1

        if new_s != s or new_e != e:
            if is_valid_offsets(text, new_s, new_e) and new_s < new_e:
                new_text = text[new_s:new_e]
                out.append(
                    Mention(
                        span_id=f"{m.span_id}__tokedge",
                        span=Span(
                            text=new_text,
                            start=new_s,
                            end=new_e,
                            provenance={
                                "op": "TOKEN_EDGE_EXPAND",
                                "source_span_id": m.span_id,
                            },
                        ),
                        ent_type=m.ent_type,
                        confidence=m.confidence,
                        evidence=_dedupe_evidence(
                            _merge_evidence(
                                m.evidence,
                                [Evidence(quote=new_text, start=new_s, end=new_e)],
                            )
                        ),
                        rationale=f"{m.rationale}|pipeline:token_edge_expand",
                    )
                )
                changed += 1
                continue
        out.append(m)
    return out, changed


def _trim_left_noise_tokens(
    text: str,
    mentions: list[Mention],
) -> tuple[list[Mention], int]:
    out: list[Mention] = []
    trimmed = 0
    for mention in mentions:
        span_text = mention.span.text
        tokens = list(re.finditer(r"[A-Za-z0-9]+", span_text))
        if len(tokens) < 2:
            out.append(mention)
            continue

        first_raw = tokens[0].group(0)
        second_raw = tokens[1].group(0)
        first_is_numeric_prefix = first_raw.isdigit() and len(first_raw) <= 2
        first_is_lower_prefix = (
            first_raw.isalpha()
            and first_raw.islower()
            and 3 <= len(first_raw) <= 8
        )
        if not (first_is_numeric_prefix or first_is_lower_prefix):
            out.append(mention)
            continue

        suffix_text = span_text[tokens[1].start() :]
        second_has_signal = any(ch.isupper() for ch in second_raw) or any(ch.isdigit() for ch in second_raw)
        suffix_has_title_token = bool(re.search(r"\b[A-Z][a-z]{2,}\b", suffix_text))
        if not (second_has_signal or _contains_symbolic_anchor(suffix_text) or suffix_has_title_token):
            out.append(mention)
            continue

        rel_start = tokens[1].start()
        new_start = mention.span.start + rel_start
        new_end = mention.span.end
        if not is_valid_offsets(text, new_start, new_end) or new_start >= new_end:
            out.append(mention)
            continue
        merged_text = text[new_start:new_end]
        if not _looks_entity_like_merged_mention(merged_text):
            out.append(mention)
            continue

        ev = _merge_evidence(
            mention.evidence,
            [Evidence(quote=merged_text, start=new_start, end=new_end)],
        )
        out.append(
            Mention(
                span_id=f"{mention.span_id}__ltrim",
                span=Span(
                    text=merged_text,
                    start=new_start,
                    end=new_end,
                    provenance={
                        "op": "LEFT_NOISE_TRIM",
                        "source_span_id": mention.span_id,
                        "dropped_prefix": first_raw.lower(),
                    },
                ),
                ent_type=mention.ent_type,
                confidence=mention.confidence,
                evidence=_dedupe_evidence(ev),
                rationale=f"{mention.rationale}|pipeline:left_noise_trim",
            )
        )
        trimmed += 1
    return out, trimmed


def _merge_coordinated_modifier_mentions(
    text: str,
    mentions: list[Mention],
) -> tuple[list[Mention], int]:
    _ = text
    return mentions, 0


def _expand_mentions_with_hyphen_suffix(
    text: str,
    mentions: list[Mention],
) -> tuple[list[Mention], int]:
    out: list[Mention] = []
    expanded = 0
    for mention in mentions:
        new_end = _find_hyphen_suffix_end(text, mention.span.start, mention.span.end)
        if new_end is None or new_end <= mention.span.end:
            out.append(mention)
            continue
        merged_text = text[mention.span.start : new_end]
        if len(merged_text) > 80:
            out.append(mention)
            continue
        if not _looks_entity_like_merged_mention(merged_text):
            out.append(mention)
            continue

        evidence = _merge_evidence(mention.evidence, [])
        evidence.append(
            Evidence(quote=merged_text, start=mention.span.start, end=new_end)
        )
        out.append(
            Mention(
                span_id=f"{mention.span_id}__hyexp",
                span=Span(
                    text=merged_text,
                    start=mention.span.start,
                    end=new_end,
                    provenance={
                        "op": "HYPHEN_RIGHT_EXPAND",
                        "source_span_id": mention.span_id,
                    },
                ),
                ent_type=mention.ent_type,
                confidence=mention.confidence,
                evidence=_dedupe_evidence(evidence),
                rationale=f"{mention.rationale}|pipeline:hyphen_expand",
            )
        )
        expanded += 1
    return out, expanded


def _expand_mentions_with_descriptors(
    text: str,
    mentions: list[Mention],
    descriptor_terms: list[str],
    left_modifiers: list[str],
) -> tuple[list[Mention], int]:
    out: list[Mention] = []
    expanded = 0
    descriptor_term_set = {t.lower() for t in descriptor_terms if t}
    if not descriptor_term_set:
        return mentions, 0

    for mention in mentions:
        out.append(mention)
        new_start, new_end = _find_descriptor_span(
            text=text,
            start=mention.span.start,
            end=mention.span.end,
            descriptor_terms=descriptor_term_set,
            left_modifiers={x.lower() for x in left_modifiers if x},
        )
        if new_start == mention.span.start and new_end == mention.span.end:
            continue
        if not is_valid_offsets(text, new_start, new_end) or new_start >= new_end:
            continue
        merged_text = text[new_start:new_end]
        if len(merged_text) > 140:
            continue
        if not _looks_descriptor_expansion_candidate(merged_text):
            continue

        evidence = _merge_evidence(mention.evidence, [])
        evidence.append(Evidence(quote=merged_text, start=new_start, end=new_end))
        out.append(
            Mention(
                span_id=f"{mention.span_id}__bioexp",
                span=Span(
                    text=merged_text,
                    start=new_start,
                    end=new_end,
                    provenance={
                        "op": "DESCRIPTOR_EXPAND",
                        "source_span_id": mention.span_id,
                    },
                ),
                ent_type=mention.ent_type,
                confidence=min(0.99, mention.confidence + 0.01),
                evidence=_dedupe_evidence(evidence),
                rationale=f"{mention.rationale}|pipeline:descriptor_expand",
            )
        )
        expanded += 1
    return out, expanded


def _find_descriptor_span(
    text: str,
    start: int,
    end: int,
    descriptor_terms: set[str],
    left_modifiers: set[str],
) -> tuple[int, int]:
    if not is_valid_offsets(text, start, end) or start >= end:
        return start, end

    end = _extend_attached_alnum_suffix(text=text, start=start, end=end)

    mention_text = text[start:end].strip().lower()
    has_descriptor_in_mention = any(
        tok in descriptor_terms for tok in re.findall(r"[A-Za-z0-9]+", mention_text)
    )

    best_end = end
    scan_end = min(len(text), end + 80)
    cursor = end
    descriptor_hits = 1 if has_descriptor_in_mention else 0
    bridge_hits = 0
    while cursor < scan_end and descriptor_hits < 3:
        next_word = _word_after(text, cursor)
        if next_word is None:
            break
        token, tok_start, tok_end = next_word
        bridge = text[cursor:tok_start]
        if any(ch in ".;:[]{}" for ch in bridge):
            break
        lower = token.lower()
        if lower in descriptor_terms:
            best_end = tok_end
            cursor = tok_end
            descriptor_hits += 1
            bridge_hits = 0
            continue
        if (
            descriptor_hits == 0
            and bridge_hits < 2
            and _is_descriptor_bridge_token(token=token, lower=lower)
        ):
            cursor = tok_end
            bridge_hits += 1
            continue
        break

    if best_end > end:
        best_end = _extend_descriptor_tail(
            text=text,
            start=start,
            end=best_end,
            descriptor_terms=descriptor_terms,
        )

    best_start = _expand_left_descriptor_modifiers(
        text=text,
        start=start,
        end=best_end,
        left_modifiers=left_modifiers,
        allow_generic_lowercase_prefix=(best_end > end or has_descriptor_in_mention),
        anchor_start=start,
        anchor_end=end,
    )
    return best_start, best_end


def _extend_attached_alnum_suffix(text: str, start: int, end: int) -> int:
    if not is_valid_offsets(text, start, end) or start >= end:
        return end
    if end >= len(text):
        return end

    prev = text[end - 1]
    nxt = text[end]
    if not (
        (prev.isdigit() and nxt.isalpha())
        or (prev.isalpha() and nxt.isdigit())
    ):
        return end

    i = end
    while i < len(text) and (text[i].isalnum() or text[i] in {"-", "_"}):
        i += 1
    if is_valid_offsets(text, start, i) and i > end:
        return i
    return end


def _extend_descriptor_tail(
    text: str,
    start: int,
    end: int,
    descriptor_terms: set[str],
) -> int:
    n = len(text)
    i = end
    steps = 0
    while i < n and steps < 3:
        j = i
        while j < n and text[j].isspace():
            j += 1
        if j >= n:
            break
        if text[j] in {".", ";", ":"}:
            break
        k = j
        while k < n and (text[k].isalnum() or text[k] in {"-", "_"}):
            k += 1
        token = text[j:k].lower()
        if not token:
            break
        if token in descriptor_terms:
            i = k
            steps += 1
            continue
        if token in {"and", "or"}:
            i = k
            steps += 1
            continue
        break
    return i if i > end else end


def _expand_left_descriptor_modifiers(
    text: str,
    start: int,
    end: int,
    left_modifiers: set[str],
    allow_generic_lowercase_prefix: bool = False,
    anchor_start: int | None = None,
    anchor_end: int | None = None,
) -> int:
    if end <= start:
        return start

    new_start = start
    anchor_s = start if anchor_start is None else anchor_start
    anchor_e = end if anchor_end is None else anchor_end
    anchor_text = text[anchor_s:anchor_e]
    for _ in range(2):
        prev = _word_before(text, new_start)
        if prev is None:
            break
        token, token_start, token_end = prev
        gap = text[token_end:new_start]
        if any(ch not in {" ", "\t"} for ch in gap):
            break
        lower = token.lower()
        if lower in left_modifiers:
            new_start = token_start
            continue
        if (
            _contains_symbolic_anchor(anchor_text)
            and _is_titlecase_word(token)
        ):
            new_start = token_start
            break
        if (
            allow_generic_lowercase_prefix
            and token.isalpha()
            and token.islower()
            and 4 <= len(token) <= 16
            and _contains_symbolic_anchor(anchor_text)
            and not re.fullmatch(r"[ivxlcdm]+", lower)
        ):
            new_start = token_start
            break
        break

    return new_start


def _looks_descriptor_expansion_candidate(text: str) -> bool:
    return _looks_entity_like_proposal(text, rationale="")


def _find_hyphen_suffix_end(text: str, start: int, end: int) -> int | None:
    n = len(text)
    i = end
    while i < n and text[i].isspace():
        i += 1
    if i >= n or text[i] != "-":
        return None
    i += 1
    while i < n and text[i].isspace():
        i += 1
    if i >= n or not text[i].isalnum():
        return None

    token_ends: list[int] = []
    tokens: list[str] = []

    first_start = i
    while i < n and (text[i].isalnum() or text[i] == "-"):
        i += 1
    if i <= first_start:
        return None
    tokens.append(text[first_start:i])
    token_ends.append(i)

    first_token = tokens[0]
    left_token = text[start:end].strip()
    if _is_short_symbol(left_token) and _is_short_symbol(first_token):
        return None
    if first_token.isdigit():
        return token_ends[0]

    for _ in range(3):
        j = i
        while j < n and text[j].isspace():
            j += 1
        if j >= n or not text[j].isalnum():
            break
        k = j
        while k < n and (text[k].isalnum() or text[k] == "-"):
            k += 1
        tokens.append(text[j:k])
        token_ends.append(k)
        i = k

    mixed_idx = -1
    for idx, tok in enumerate(tokens):
        has_alpha = any(ch.isalpha() for ch in tok)
        has_digit = any(ch.isdigit() for ch in tok)
        if has_alpha and has_digit:
            mixed_idx = idx
            break
    if mixed_idx >= 0:
        new_end = token_ends[mixed_idx]
        if is_valid_offsets(text, start, new_end):
            return new_end

    digit_idx = -1
    for idx, tok in enumerate(tokens):
        if any(ch.isdigit() for ch in tok):
            digit_idx = idx
            break
    if digit_idx >= 0:
        preceding = tokens[:digit_idx]
        if any(t.isalpha() and t.islower() for t in preceding):
            return None
        new_end = token_ends[digit_idx]
        if is_valid_offsets(text, start, new_end):
            return new_end

    alpha_idx = next(
        (
            idx
            for idx, tok in enumerate(tokens)
            if tok.isalpha() and len(tok) >= 3
        ),
        -1,
    )
    if 0 <= alpha_idx <= 1:
        end_idx = alpha_idx
        step = 0
        while (
            end_idx + 1 < len(tokens)
            and step < 1
            and tokens[end_idx + 1].isalpha()
            and len(tokens[end_idx + 1]) >= 3
        ):
            end_idx += 1
            step += 1
        new_end = token_ends[end_idx]
        if is_valid_offsets(text, start, new_end):
            return new_end

    return None


def _contains_span(container: Span, inner: Span) -> bool:
    return container.start <= inner.start and inner.end <= container.end


def _merge_evidence(left: list[Evidence], right: list[Evidence]) -> list[Evidence]:
    return [Evidence(quote=e.quote, start=e.start, end=e.end) for e in left + right]


def _dedupe_evidence(evidence: list[Evidence]) -> list[Evidence]:
    seen: set[tuple[str, int, int]] = set()
    out: list[Evidence] = []
    for ev in evidence:
        key = (ev.quote, ev.start, ev.end)
        if key in seen:
            continue
        seen.add(key)
        out.append(ev)
    return out


def _is_descriptor_rich(text: str) -> bool:
    tokens = re.findall(r"[A-Za-z0-9]+", text)
    if len(tokens) >= 3:
        return True
    if any(
        any(ch.isalpha() for ch in tok) and any(ch.isdigit() for ch in tok)
        for tok in tokens
    ):
        return True
    return any(tok.isalpha() and len(tok) >= 8 for tok in tokens)


def _split_symbolic_chain_mentions(
    text: str,
    mentions: list[Mention],
) -> tuple[list[Mention], int]:
    out: list[Mention] = []
    split_count = 0
    for mention in mentions:
        split_parts = _extract_symbolic_chain_parts(mention)
        if len(split_parts) < 2:
            out.append(mention)
            continue
        split_count += 1
        for idx, (s, e) in enumerate(split_parts):
            part_text = text[s:e]
            out.append(
                Mention(
                    span_id=f"{mention.span_id}__split__{idx}",
                    span=Span(
                        text=part_text,
                        start=s,
                        end=e,
                        provenance={
                            "op": "SYMBOLIC_CHAIN_SPLIT",
                            "source_span_id": mention.span_id,
                        },
                    ),
                    ent_type=mention.ent_type,
                    confidence=mention.confidence,
                    evidence=_dedupe_evidence(
                        _merge_evidence(
                            mention.evidence,
                            [Evidence(quote=part_text, start=s, end=e)],
                        )
                    ),
                    rationale=f"{mention.rationale}|pipeline:symbolic_chain_split",
                )
            )
    return out, split_count


def _split_slash_coordinated_mentions(
    text: str,
    mentions: list[Mention],
) -> tuple[list[Mention], int]:
    out: list[Mention] = []
    split_count = 0
    for mention in mentions:
        span_text = mention.span.text
        if "/" not in span_text:
            out.append(mention)
            continue

        slash_idx = span_text.find("/")
        left_raw = span_text[:slash_idx].strip()
        right_raw = span_text[slash_idx + 1 :].strip()
        right_core = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", right_raw)
        left_core = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", left_raw)

        if not left_core or not right_core:
            out.append(mention)
            continue
        if not _looks_slash_part_entity_like(left_core):
            out.append(mention)
            continue
        if not _looks_slash_part_entity_like(right_core):
            out.append(mention)
            continue

        left_rel = span_text.find(left_core)
        right_rel = span_text.find(right_core, slash_idx + 1)
        if left_rel < 0 or right_rel < 0:
            out.append(mention)
            continue

        left_start = mention.span.start + left_rel
        left_end = left_start + len(left_core)
        right_start = mention.span.start + right_rel
        right_end = right_start + len(right_core)
        if not is_valid_offsets(text, left_start, left_end):
            out.append(mention)
            continue
        if not is_valid_offsets(text, right_start, right_end):
            out.append(mention)
            continue

        split_count += 1
        for idx, (s, e, part_text) in enumerate(
            [
                (left_start, left_end, text[left_start:left_end]),
                (right_start, right_end, text[right_start:right_end]),
            ]
        ):
            out.append(
                Mention(
                    span_id=f"{mention.span_id}__slash__{idx}",
                    span=Span(
                        text=part_text,
                        start=s,
                        end=e,
                        provenance={
                            "op": "SLASH_SPLIT",
                            "source_span_id": mention.span_id,
                        },
                    ),
                    ent_type=mention.ent_type,
                    confidence=mention.confidence,
                    evidence=_dedupe_evidence(
                        _merge_evidence(
                            mention.evidence,
                            [Evidence(quote=part_text, start=s, end=e)],
                        )
                    ),
                    rationale=f"{mention.rationale}|pipeline:slash_split",
                )
            )
    return out, split_count


def _looks_slash_part_entity_like(part: str) -> bool:
    tokens = re.findall(r"[A-Za-z0-9]+", part)
    if not tokens:
        return False
    if all(t.isdigit() for t in tokens):
        return False
    if len(tokens) > 6:
        return False
    if all(re.fullmatch(r"[ivxlcdm]+", t.lower()) for t in tokens):
        return False
    has_upper = any(any(ch.isupper() for ch in t) for t in tokens)
    has_digit = any(any(ch.isdigit() for ch in t) for t in tokens)
    if has_upper or has_digit:
        return True
    return False


def _resolve_prompt_path(repo_root: Path, raw_path: Any) -> Path:
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = repo_root / path
    return path


def _rescue_symbolic_parts_from_hypotheses(
    text: str,
    final_mentions: list[Mention],
    source_mentions: list[Mention],
    min_confidence: float = 0.85,
) -> tuple[list[Mention], dict[str, Any]]:
    if not source_mentions:
        return final_mentions, {"enabled": True, "added_count": 0, "added_span_ids": []}

    out = list(final_mentions)
    existing = {(m.span.start, m.span.end, m.ent_type) for m in out}
    added_ids: list[str] = []
    added = 0

    for src in source_mentions:
        if src.confidence < min_confidence:
            continue
        parts = _extract_symbolic_chain_parts(src)
        if len(parts) < 2:
            continue
        for idx, (s, e) in enumerate(parts):
            if not is_valid_offsets(text, s, e) or s >= e:
                continue
            part_text = text[s:e]
            if not _is_rescuable_symbol_part(part_text):
                continue
            key = (s, e, src.ent_type)
            if key in existing:
                continue
            mention = Mention(
                span_id=f"{src.span_id}__rescue_split__{idx}",
                span=Span(
                    text=part_text,
                    start=s,
                    end=e,
                    provenance={
                        "op": "RESCUE_SPLIT",
                        "source_span_id": src.span_id,
                    },
                ),
                ent_type=src.ent_type,
                confidence=min(0.92, src.confidence),
                evidence=_dedupe_evidence(
                    _merge_evidence(
                        src.evidence,
                        [Evidence(quote=part_text, start=s, end=e)],
                    )
                ),
                rationale=f"{src.rationale}|pipeline:rescue_symbolic_split",
            )
            out.append(mention)
            existing.add(key)
            added += 1
            added_ids.append(mention.span_id)

    out, _ = _dedupe_mentions_by_offset_and_type(out)
    out = sorted(out, key=lambda m: (m.span.start, m.span.end, m.span_id))
    return out, {
        "enabled": True,
        "added_count": added,
        "added_span_ids": added_ids,
    }


def _extract_parenthetical_alias_mentions(
    text: str,
    mentions: list[Mention],
) -> tuple[list[Mention], dict[str, Any]]:
    if not mentions:
        return mentions, {"enabled": True, "added_count": 0, "added_span_ids": []}

    out = list(mentions)
    existing = {(m.span.start, m.span.end, m.ent_type) for m in out}
    added_ids: list[str] = []
    added = 0

    alias_pattern = re.compile(r"[\(\[]\s*([A-Za-z]{2,8})\s*[\)\]]")
    for mention in mentions:
        for idx, m in enumerate(alias_pattern.finditer(mention.span.text)):
            alias = m.group(1)
            s = mention.span.start + m.start(1)
            e = mention.span.start + m.end(1)
            if not is_valid_offsets(text, s, e) or s >= e:
                continue
            if text[s:e] != alias:
                continue
            key = (s, e, mention.ent_type)
            if key in existing:
                continue
            if not _is_rescuable_alias_token(alias=alias, parent_text=mention.span.text):
                continue
            rescued = Mention(
                span_id=f"{mention.span_id}__alias__{idx}",
                span=Span(
                    text=alias,
                    start=s,
                    end=e,
                    provenance={
                        "op": "ALIAS_EXTRACT",
                        "source_span_id": mention.span_id,
                    },
                ),
                ent_type=mention.ent_type,
                confidence=min(mention.confidence, 0.95),
                evidence=_dedupe_evidence(
                    _merge_evidence(
                        mention.evidence,
                        [Evidence(quote=alias, start=s, end=e)],
                    )
                ),
                rationale=f"{mention.rationale}|pipeline:alias_extract",
            )
            out.append(rescued)
            existing.add(key)
            added += 1
            added_ids.append(rescued.span_id)

    out, _ = _dedupe_mentions_by_offset_and_type(out)
    out = sorted(out, key=lambda m: (m.span.start, m.span.end, m.span_id))
    return out, {
        "enabled": True,
        "added_count": added,
        "added_span_ids": added_ids,
    }


def _extract_symbolic_chain_parts(mention: Mention) -> list[tuple[int, int]]:
    text = mention.span.text
    # Match mentions like "UCP1 - CAT" / "X - Y - Z".
    if "-" not in text:
        return []
    if _is_compact_lowercase_hyphen_symbol(text):
        return []
    token_matches = list(re.finditer(r"[A-Za-z0-9]+", text))
    if len(token_matches) < 2:
        return []

    parts: list[tuple[int, int]] = []
    for m in token_matches:
        tok = m.group(0)
        if tok.isalpha() and tok.islower() and len(tok) >= 5:
            break
        if len(tok) == 1 and not any(ch.isdigit() for ch in tok):
            return []
        if len(tok) > 6:
            return []
        if not any(ch.isalpha() for ch in tok):
            return []
        if any(ch.islower() for ch in tok) and not any(ch.isdigit() for ch in tok):
            return []
        if any(ch.islower() for ch in tok) and any(ch.isupper() for ch in tok):
            return []
        if tok.islower() and not any(ch.isdigit() for ch in tok):
            return []
        s = mention.span.start + m.start()
        e = mention.span.start + m.end()
        parts.append((s, e))

    if len(parts) < 2:
        return []

    between = text[token_matches[0].end() : token_matches[min(len(parts) - 1, len(token_matches)-1)].start()]
    if "-" not in between:
        return []
    return parts


def _is_rescuable_symbol_part(text: str) -> bool:
    token = text.strip()
    if not token:
        return False
    if len(token) > 12:
        return False
    has_alpha = any(ch.isalpha() for ch in token)
    if not has_alpha:
        return False
    has_upper = any(ch.isupper() for ch in token)
    has_digit = any(ch.isdigit() for ch in token)
    return has_upper or has_digit


def _is_rescuable_alias_token(alias: str, parent_text: str) -> bool:
    if _is_rescuable_symbol_part(alias):
        return True
    tok = alias.strip()
    if not tok.islower():
        return False
    if len(tok) < 2 or len(tok) > 6:
        return False
    parent = parent_text.strip()
    has_parent_symbol = any(ch.isdigit() for ch in parent) and any(ch.isalpha() for ch in parent)
    return has_parent_symbol


def _should_merge_adjacent_space_tokens(left_text: str, right_text: str) -> bool:
    left = left_text.strip()
    right = right_text.strip()
    if not left or not right:
        return False
    if len(left.split()) != 1 or len(right.split()) != 1:
        return False
    if len(left) > 12 or len(right) > 12:
        return False

    left_symbol = bool(re.search(r"[A-Z]", left)) and bool(re.search(r"\d", left))
    right_symbol = bool(re.search(r"[A-Z]", right)) and bool(re.search(r"\d", right))
    mixed_case_pair = left.isupper() and (right.isupper() or right_symbol)
    return left_symbol or right_symbol or mixed_case_pair


def _allow_hyphen_symbol_merge(left_text: str, right_text: str, merged_text: str) -> bool:
    left = left_text.strip()
    right = right_text.strip()
    left_symbol = bool(re.search(r"[A-Za-z]", left)) and bool(re.search(r"\d", left))
    right_symbol = bool(re.search(r"[A-Za-z]", right)) and bool(re.search(r"\d", right))
    if left_symbol and right_symbol and len(left) <= 6 and len(right) <= 6:
        return False

    if left.isupper() and right.isupper() and len(left) <= 6 and len(right) <= 6:
        return False

    return _looks_entity_like_merged_mention(merged_text)


def _looks_entity_like_merged_mention(text: str) -> bool:
    return _looks_entity_like_proposal(text, rationale="")


def _should_preserve_symbolic_submention(anchor: Mention, mention: Mention) -> bool:
    short = mention.span.text.strip()
    long = anchor.span.text.strip()
    if not short or not long or len(long) <= len(short):
        return False

    if _is_compact_symbol_token(short) and long.startswith(short):
        tail = long[len(short) :].strip()
        if not tail:
            return False
        tail_tokens = [t.lower() for t in re.findall(r"[A-Za-z]+", tail)]
        if not tail_tokens:
            return False

        if len(tail_tokens) <= 2 and all(2 <= len(tok) <= 10 for tok in tail_tokens):
            return True

        # Keep base symbols when the longer form only appends a parenthetical alias.
        compact_tail = re.sub(r"\s+", " ", tail).strip()
        if re.fullmatch(r"[\(\[]\s*[A-Za-z0-9]{1,8}\s*[\)\]]", compact_tail):
            return True

    # Preserve symbolic/hyphenated submentions inside longer construct-style phrases.
    if short.lower() in long.lower():
        short_tokens = re.findall(r"[A-Za-z0-9]+", short)
        if short_tokens and len(short_tokens) <= 3:
            has_symbolic_signal = any(
                any(ch.isupper() for ch in tok) or any(ch.isdigit() for ch in tok)
                for tok in short_tokens
            ) or ("-" in short and len(short_tokens) >= 2)
            if has_symbolic_signal:
                long_tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9]+", long)]
                if len(long_tokens) >= len(short_tokens) + 2 or re.search(r"[\(\)\[\]-]", long):
                    return True
    return False


def _is_compact_symbol_token(text: str) -> bool:
    token = text.strip()
    if not token or " " in token:
        return False
    if len(token) > 16:
        return False
    has_alpha = any(ch.isalpha() for ch in token)
    has_digit = any(ch.isdigit() for ch in token)
    has_upper = any(ch.isupper() for ch in token)
    return has_alpha and (has_digit or has_upper)


def _looks_symbolic_anchor(text: str) -> bool:
    token = text.strip()
    if not token:
        return False
    if " " in token:
        return False
    return _is_compact_symbol_token(token)


def _contains_symbolic_anchor(text: str) -> bool:
    tokens = re.findall(r"[A-Za-z0-9\-_]+", text)
    if not tokens:
        return False
    return any(_is_compact_symbol_token(tok) for tok in tokens)


def _is_titlecase_word(token: str) -> bool:
    t = token.strip()
    if len(t) < 2 or len(t) > 14:
        return False
    if not t[0].isupper():
        return False
    return all(ch.islower() for ch in t[1:] if ch.isalpha())


def _is_descriptor_bridge_token(token: str, lower: str) -> bool:
    _ = lower
    if re.fullmatch(r"[A-Za-z]*\d+[A-Za-z0-9]*", token):
        return True
    if "-" in token and any(ch.isalpha() for ch in token):
        return True
    if token.isalpha() and token.islower() and len(token) >= 7:
        return True
    return False


def _is_compact_lowercase_hyphen_symbol(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if not re.fullmatch(r"[A-Za-z]+\d+[A-Za-z0-9]*-[0-9]+[A-Za-z]+", compact):
        return False
    return any(ch.islower() for ch in compact)


def _inject_configured_candidate_mentions(
    text: str,
    candidate_set: CandidateSet,
    mentions: list[Mention],
    rules: list[Any],
    default_ent_type: str,
) -> tuple[list[Mention], dict[str, Any]]:
    rule_map = _compile_lexical_rules(rules, default_ent_type=default_ent_type)
    if not rule_map:
        return mentions, {"enabled": False, "added": 0, "added_span_ids": []}

    existing_offsets = {(m.span.start, m.span.end, m.ent_type) for m in mentions}
    existing_ranges = [(m.span.start, m.span.end) for m in mentions]
    out = list(mentions)
    added = 0
    added_ids: list[str] = []

    for span_id, span in candidate_set.spans.items():
        norm = span.text.strip().lower()
        rule = rule_map.get(norm)
        if rule is None:
            continue
        ent_type = str(rule["ent_type"])
        confidence = float(rule["confidence"])

        if (span.start, span.end, ent_type) in existing_offsets:
            continue
        if any(max(span.start, s) < min(span.end, e) for s, e in existing_ranges):
            continue
        if not is_valid_offsets(text, span.start, span.end) or span.start >= span.end:
            continue
        mention = Mention(
            span_id=f"{span_id}__hvcand",
            span=Span(
                text=text[span.start : span.end],
                start=span.start,
                end=span.end,
                provenance={
                    "op": "LEXICAL_RULE_INJECT",
                    "source_span_id": span_id,
                    "rule": norm,
                },
            ),
            ent_type=ent_type,
            confidence=confidence,
            evidence=[Evidence(quote=text[span.start : span.end], start=span.start, end=span.end)],
            rationale=f"pipeline:lexical_injection:{span_id}",
        )
        out.append(mention)
        existing_offsets.add((span.start, span.end, ent_type))
        existing_ranges.append((span.start, span.end))
        added += 1
        added_ids.append(mention.span_id)

    out, _ = _dedupe_mentions_by_offset_and_type(out)
    out = sorted(out, key=lambda m: (m.span.start, m.span.end, m.span_id))
    trace = {
        "enabled": True,
        "added": added,
        "added_span_ids": added_ids,
    }
    return out, trace


def _drop_blocklisted_mentions(
    mentions: list[Mention],
    blocked_terms: set[str] | None = None,
) -> tuple[list[Mention], dict[str, Any]]:
    blocked = {x.strip().lower() for x in (blocked_terms or set()) if x and x.strip()}
    if not blocked:
        return mentions, {"enabled": False, "dropped": 0, "dropped_span_ids": []}

    out: list[Mention] = []
    dropped: list[str] = []
    for m in mentions:
        norm = m.span.text.strip().lower()
        if norm in blocked:
            dropped.append(m.span_id)
            continue
        out.append(m)
    return out, {
        "enabled": True,
        "dropped": len(dropped),
        "dropped_span_ids": dropped,
    }


def _drop_low_information_mentions(
    mentions: list[Mention],
    titlecase_confidence_threshold: float = 0.9,
    descriptor_confidence_threshold: float = 0.9,
) -> tuple[list[Mention], dict[str, Any]]:
    out: list[Mention] = []
    dropped_ids: list[str] = []

    for m in mentions:
        text = m.span.text.strip()
        tokens = re.findall(r"[A-Za-z0-9]+", text)
        has_upper = any(any(ch.isupper() for ch in t) for t in tokens)
        has_digit = any(any(ch.isdigit() for ch in t) for t in tokens)

        if text and re.fullmatch(r"[IVXLCM]+", text):
            dropped_ids.append(m.span_id)
            continue

        if (
            len(tokens) == 1
            and tokens[0][:1].isupper()
            and tokens[0][1:].islower()
            and not has_digit
            and m.confidence < titlecase_confidence_threshold
        ):
            dropped_ids.append(m.span_id)
            continue

        if (
            tokens
            and not has_upper
            and not has_digit
            and m.confidence <= descriptor_confidence_threshold
            and all(t.isalpha() for t in tokens)
            and len(tokens) <= 2
            and (sum(len(t) for t in tokens) / len(tokens)) <= 6.0
        ):
            dropped_ids.append(m.span_id)
            continue

        out.append(m)

    return out, {
        "enabled": True,
        "dropped": len(dropped_ids),
        "dropped_span_ids": dropped_ids,
    }

def _promote_mentions_to_candidate_superspans(
    text: str,
    mentions: list[Mention],
    candidate_set: CandidateSet,
    schema: Any,
    max_expand_chars: int = 40,
) -> tuple[list[Mention], dict[str, Any]]:
    if not mentions:
        return mentions, {"enabled": True, "promoted_count": 0, "promoted_span_ids": []}
    if not candidate_set.spans:
        return mentions, {"enabled": True, "promoted_count": 0, "promoted_span_ids": []}

    descriptor_terms = set(_schema_descriptor_terms(schema))
    candidates = sorted(candidate_set.spans.values(), key=lambda s: (s.start, s.end))
    stop_prefixes = {"the", "a", "an", "many", "several", "various", "some"}

    promoted_ids: list[str] = []
    out: list[Mention] = []
    for mention in mentions:
        best_span: Span | None = None
        mention_len = mention.span.end - mention.span.start
        for cand in candidates:
            if cand.start > mention.span.start or mention.span.end > cand.end:
                continue
            if cand.start == mention.span.start and cand.end == mention.span.end:
                continue
            cand_len = cand.end - cand.start
            if cand_len <= mention_len:
                continue
            if (cand_len - mention_len) > max(0, max_expand_chars):
                continue
            if not is_valid_offsets(text, cand.start, cand.end):
                continue

            cand_text = text[cand.start : cand.end]
            cand_tokens = re.findall(r"[A-Za-z0-9]+", cand_text)
            if not cand_tokens or len(cand_tokens) > 8:
                continue
            if cand_tokens[0].lower() in stop_prefixes:
                continue
            if descriptor_terms and not any(t.lower() in descriptor_terms for t in cand_tokens):
                continue
            if descriptor_terms:
                tail = [cand_tokens[-1].lower()]
                if len(cand_tokens) >= 2:
                    tail.append(cand_tokens[-2].lower())
                if not any(t in descriptor_terms for t in tail):
                    continue
            if not _is_candidate_recall_entity_like(cand_text):
                continue

            if best_span is None or (cand.end - cand.start) > (best_span.end - best_span.start):
                best_span = cand

        if best_span is None:
            out.append(mention)
            continue

        promoted_text = text[best_span.start : best_span.end]
        promoted = Mention(
            span_id=f"{mention.span_id}__sup",
            span=Span(
                text=promoted_text,
                start=best_span.start,
                end=best_span.end,
                provenance={
                    "op": "CANDIDATE_SUPERSPAN_PROMOTION",
                    "source_span_id": mention.span_id,
                },
            ),
            ent_type=mention.ent_type,
            confidence=mention.confidence,
            evidence=_dedupe_evidence(
                _merge_evidence(
                    mention.evidence,
                    [Evidence(quote=promoted_text, start=best_span.start, end=best_span.end)],
                )
            ),
            rationale=f"{mention.rationale}|pipeline:candidate_superspan",
        )
        out.append(promoted)
        promoted_ids.append(promoted.span_id)

    out, _ = _dedupe_mentions_by_offset_and_type(out)
    out = sorted(out, key=lambda m: (m.span.start, m.span.end, m.span_id))
    return out, {
        "enabled": True,
        "promoted_count": len(promoted_ids),
        "promoted_span_ids": promoted_ids,
    }


def _align_mentions_to_candidate_boundaries(
    text: str,
    mentions: list[Mention],
    candidate_set: CandidateSet,
    schema: Any,
    max_expand_chars: int = 64,
    max_tokens: int = 14,
    min_score_gain: float = 0.08,
) -> tuple[list[Mention], dict[str, Any]]:
    if not mentions:
        return mentions, {"enabled": True, "aligned_count": 0, "aligned_span_ids": []}
    if not candidate_set.spans:
        return mentions, {"enabled": True, "aligned_count": 0, "aligned_span_ids": []}

    descriptor_terms = set(_schema_descriptor_terms(schema))
    stop_prefixes = {"the", "a", "an", "many", "several", "various", "some", "this", "that"}
    candidates = sorted(candidate_set.spans.items(), key=lambda x: (x[1].start, x[1].end))

    aligned_ids: list[str] = []
    out: list[Mention] = []

    for mention in mentions:
        base_span = mention.span
        base_len = max(1, base_span.end - base_span.start)
        best: tuple[str, Span, str, float] | None = None

        for cand_id, cand in candidates:
            cand_len = cand.end - cand.start
            if cand_len <= 0:
                continue
            if not _span_offsets_overlap(base_span.start, base_span.end, cand.start, cand.end):
                continue
            if abs(cand_len - base_len) > max(0, max_expand_chars):
                continue

            cand_text = text[cand.start : cand.end]
            cand_tokens = re.findall(r"[A-Za-z0-9]+", cand_text)
            if not cand_tokens or len(cand_tokens) > max(1, max_tokens):
                continue
            if cand_tokens[0].lower() in stop_prefixes and cand.start < base_span.start:
                continue
            if not _boundary_candidate_compatible(base_span.text, cand_text):
                continue

            score = _boundary_alignment_score(
                mention_start=base_span.start,
                mention_end=base_span.end,
                mention_text=base_span.text,
                cand_start=cand.start,
                cand_end=cand.end,
                cand_text=cand_text,
                descriptor_terms=descriptor_terms,
            )
            if best is None or score > best[3]:
                best = (cand_id, cand, cand_text, score)

        if best is None:
            out.append(mention)
            continue

        _, best_span, best_text, best_score = best
        if best_score < float(min_score_gain):
            out.append(mention)
            continue
        if best_span.start == base_span.start and best_span.end == base_span.end:
            out.append(mention)
            continue

        aligned = Mention(
            span_id=f"{mention.span_id}__baln",
            span=Span(
                text=best_text,
                start=best_span.start,
                end=best_span.end,
                provenance={
                    "op": "CANDIDATE_BOUNDARY_ALIGNMENT",
                    "source_span_id": mention.span_id,
                },
            ),
            ent_type=mention.ent_type,
            confidence=mention.confidence,
            evidence=_dedupe_evidence(
                _merge_evidence(
                    mention.evidence,
                    [Evidence(quote=best_text, start=best_span.start, end=best_span.end)],
                )
            ),
            rationale=f"{mention.rationale}|pipeline:candidate_boundary_alignment",
        )
        out.append(aligned)
        aligned_ids.append(aligned.span_id)

    out, _ = _dedupe_mentions_by_offset_and_type(out)
    out = sorted(out, key=lambda m: (m.span.start, m.span.end, m.span_id))
    return out, {"enabled": True, "aligned_count": len(aligned_ids), "aligned_span_ids": aligned_ids}


def _span_offsets_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)


def _span_offsets_iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    if inter <= 0:
        return 0.0
    union = max(a_end, b_end) - min(a_start, b_start)
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def _boundary_candidate_compatible(mention_text: str, cand_text: str) -> bool:
    mention_tokens = re.findall(r"[A-Za-z0-9]+", mention_text.lower())
    cand_tokens = re.findall(r"[A-Za-z0-9]+", cand_text.lower())
    if not mention_tokens or not cand_tokens:
        return False

    mention_norm = re.sub(r"[^A-Za-z0-9]+", "", mention_text).lower()
    cand_norm = re.sub(r"[^A-Za-z0-9]+", "", cand_text).lower()
    if mention_norm and cand_norm and (mention_norm in cand_norm or cand_norm in mention_norm):
        return True

    mention_set = set(mention_tokens)
    cand_set = set(cand_tokens)
    overlap = len(mention_set.intersection(cand_set))
    if overlap <= 0:
        return False
    overlap_ratio = overlap / float(max(1, min(len(mention_set), len(cand_set))))
    return overlap_ratio >= 0.5


def _boundary_alignment_score(
    mention_start: int,
    mention_end: int,
    mention_text: str,
    cand_start: int,
    cand_end: int,
    cand_text: str,
    descriptor_terms: set[str],
) -> float:
    iou = _span_offsets_iou(mention_start, mention_end, cand_start, cand_end)
    mention_set = set(re.findall(r"[A-Za-z0-9]+", mention_text.lower()))
    cand_tokens = re.findall(r"[A-Za-z0-9]+", cand_text.lower())
    cand_set = set(cand_tokens)
    inter = len(mention_set.intersection(cand_set))
    union = len(mention_set.union(cand_set))
    token_jaccard = (inter / float(union)) if union else 0.0

    contains_bonus = 0.0
    if cand_start <= mention_start and mention_end <= cand_end:
        contains_bonus += 0.14
    if mention_start <= cand_start and cand_end <= mention_end:
        contains_bonus += 0.06

    tail_bonus = 0.0
    if cand_tokens and descriptor_terms:
        tail = cand_tokens[-1]
        prev = cand_tokens[-2] if len(cand_tokens) >= 2 else ""
        if tail in descriptor_terms or prev in descriptor_terms:
            tail_bonus = 0.08

    entity_like_bonus = 0.08 if _is_candidate_recall_entity_like(cand_text) else -0.05

    cand_len = max(1, cand_end - cand_start)
    mention_len = max(1, mention_end - mention_start)
    expand_ratio = max(0.0, float(cand_len - mention_len) / float(mention_len))
    shrink_ratio = max(0.0, float(mention_len - cand_len) / float(mention_len))
    length_penalty = min(0.22, 0.10 * expand_ratio) + min(0.10, 0.08 * shrink_ratio)

    return (0.45 * iou) + (0.28 * token_jaccard) + contains_bonus + tail_bonus + entity_like_bonus - length_penalty


def _inject_candidate_recall_mentions(
    text: str,
    candidate_set: CandidateSet,
    mentions: list[Mention],
    constraints: ExpertConstraints,
    schema: Any,
    max_added: int = 10,
    min_confidence: float = 0.78,
    skip_excluded_confidence: float = 0.9,
) -> tuple[list[Mention], dict[str, Any]]:
    entity_types = list(getattr(schema, "entity_type_names", []) or [])
    if len(entity_types) != 1:
        return mentions, {
            "enabled": False,
            "reason": "multi_type_schema",
            "added_count": 0,
            "added_span_ids": [],
        }
    if not candidate_set.spans:
        return mentions, {"enabled": True, "added_count": 0, "added_span_ids": []}

    ent_type = entity_types[0]
    out = list(mentions)
    existing = {(m.span.start, m.span.end, m.ent_type) for m in out}
    existing_ranges = [(m.span.start, m.span.end) for m in out]
    added_ids: list[str] = []
    added = 0

    for span_id, span in sorted(candidate_set.spans.items(), key=lambda kv: (kv[1].start, kv[1].end)):
        if added >= max(0, max_added):
            break
        if not is_valid_offsets(text, span.start, span.end) or span.start >= span.end:
            continue
        if (span.start, span.end, ent_type) in existing:
            continue
        if any(max(span.start, s) < min(span.end, e) for s, e in existing_ranges):
            continue
        if not _is_candidate_recall_entity_like(span.text):
            continue

        per = constraints.per_span.get(span_id)
        if per is not None:
            excluded = {x for x in (per.excluded_types or [])}
            if ent_type in excluded and float(per.confidence) >= skip_excluded_confidence:
                continue

        mention = Mention(
            span_id=f"{span_id}__recall",
            span=Span(
                text=text[span.start : span.end],
                start=span.start,
                end=span.end,
                provenance={
                    "op": "CANDIDATE_RECALL_BRIDGE",
                    "source_span_id": span_id,
                },
            ),
            ent_type=ent_type,
            confidence=min(0.95, max(0.0, min_confidence)),
            evidence=[Evidence(quote=text[span.start : span.end], start=span.start, end=span.end)],
            rationale=f"pipeline:candidate_recall_bridge:{span_id}",
        )
        out.append(mention)
        existing.add((span.start, span.end, ent_type))
        existing_ranges.append((span.start, span.end))
        added += 1
        added_ids.append(mention.span_id)

    out, _ = _dedupe_mentions_by_offset_and_type(out)
    out = sorted(out, key=lambda m: (m.span.start, m.span.end, m.span_id))
    return out, {
        "enabled": True,
        "added_count": added,
        "added_span_ids": added_ids,
    }


def _restore_high_support_expert_mentions(
    text: str,
    current_mentions: list[Mention],
    expert_mentions: list[Mention],
    constraints: ExpertConstraints,
    schema: Any,
    max_added: int = 12,
    min_mention_confidence: float = 0.82,
    min_constraint_confidence: float = 0.6,
) -> tuple[list[Mention], dict[str, Any]]:
    if not current_mentions and not expert_mentions:
        return [], {"enabled": True, "added_count": 0, "added_span_ids": []}
    if not expert_mentions:
        return current_mentions, {"enabled": True, "added_count": 0, "added_span_ids": []}

    out = list(current_mentions)
    existing_exact = {(m.span.start, m.span.end, m.ent_type) for m in out}
    existing_ranges = [(m.span.start, m.span.end, m.ent_type) for m in out]

    def _overlaps_existing(m: Mention) -> bool:
        for s, e, t in existing_ranges:
            if t != m.ent_type:
                continue
            if max(s, m.span.start) < min(e, m.span.end):
                return True
        return False

    added_ids: list[str] = []
    added = 0
    descriptor_terms = set(_schema_descriptor_terms(schema))

    for m in sorted(expert_mentions, key=lambda x: x.confidence, reverse=True):
        if added >= max(0, max_added):
            break
        if m.confidence < float(min_mention_confidence):
            continue
        key = (m.span.start, m.span.end, m.ent_type)
        if key in existing_exact:
            continue
        if _overlaps_existing(m):
            continue
        if not _is_candidate_recall_entity_like(m.span.text):
            continue

        base_span_id = m.span_id.split("__", 1)[0]
        span_constraint = constraints.per_span.get(base_span_id)
        if span_constraint is None:
            continue
        if span_constraint.confidence < float(min_constraint_confidence):
            continue
        if m.ent_type in set(span_constraint.excluded_types or []):
            continue
        if m.ent_type not in set(span_constraint.candidate_types or []):
            continue

        restore_start = m.span.start
        restore_end = m.span.end
        restore_text = text[restore_start:restore_end]
        best_score = _boundary_alignment_score(
            mention_start=m.span.start,
            mention_end=m.span.end,
            mention_text=m.span.text,
            cand_start=restore_start,
            cand_end=restore_end,
            cand_text=restore_text,
            descriptor_terms=descriptor_terms,
        )
        for ev in span_constraint.evidence or []:
            ev_start = int(ev.start)
            ev_end = int(ev.end)
            if not is_valid_offsets(text, ev_start, ev_end) or ev_start >= ev_end:
                continue
            if not _span_offsets_overlap(m.span.start, m.span.end, ev_start, ev_end):
                continue
            if abs((ev_end - ev_start) - (m.span.end - m.span.start)) > 96:
                continue
            ev_text = text[ev_start:ev_end]
            if not _boundary_candidate_compatible(m.span.text, ev_text):
                continue
            ev_score = _boundary_alignment_score(
                mention_start=m.span.start,
                mention_end=m.span.end,
                mention_text=m.span.text,
                cand_start=ev_start,
                cand_end=ev_end,
                cand_text=ev_text,
                descriptor_terms=descriptor_terms,
            )
            if ev_score > best_score:
                best_score = ev_score
                restore_start, restore_end, restore_text = ev_start, ev_end, ev_text

        restored = Mention(
            span_id=f"{m.span_id}__restore",
            span=Span(
                text=restore_text,
                start=restore_start,
                end=restore_end,
                provenance={
                    "op": "EXPERT_SUPPORT_RESTORE",
                    "source_span_id": m.span_id,
                },
            ),
            ent_type=m.ent_type,
            confidence=max(m.confidence, span_constraint.confidence),
            evidence=_dedupe_evidence(
                _merge_evidence(
                    m.evidence,
                    [Evidence(quote=restore_text, start=restore_start, end=restore_end)],
                )
            ),
            rationale=f"{m.rationale}|pipeline:expert_support_restore",
        )
        out.append(restored)
        existing_exact.add((restored.span.start, restored.span.end, restored.ent_type))
        existing_ranges.append((restored.span.start, restored.span.end, restored.ent_type))
        added += 1
        added_ids.append(restored.span_id)

    out, _ = _dedupe_mentions_by_offset_and_type(out)
    out = sorted(out, key=lambda x: (x.span.start, x.span.end, x.span_id))
    return out, {"enabled": True, "added_count": added, "added_span_ids": added_ids}


def _calibrate_mentions_with_expert_evidence(
    text: str,
    mentions: list[Mention],
    constraints: ExpertConstraints,
    schema: Any,
    min_score_gain: float = 0.12,
) -> tuple[list[Mention], dict[str, Any]]:
    if not mentions:
        return mentions, {"enabled": True, "calibrated_count": 0, "calibrated_span_ids": []}
    if not constraints.per_span:
        return mentions, {"enabled": True, "calibrated_count": 0, "calibrated_span_ids": []}

    descriptor_terms = set(_schema_descriptor_terms(schema))
    out: list[Mention] = []
    calibrated_ids: list[str] = []

    for m in mentions:
        rationale_l = (m.rationale or "").lower()
        if "expert" not in rationale_l:
            out.append(m)
            continue

        base_span_id = m.span_id.split("__", 1)[0]
        span_constraint = constraints.per_span.get(base_span_id)
        if span_constraint is None or not span_constraint.evidence:
            out.append(m)
            continue

        best_start = m.span.start
        best_end = m.span.end
        best_text = text[best_start:best_end]
        best_score = _boundary_alignment_score(
            mention_start=m.span.start,
            mention_end=m.span.end,
            mention_text=m.span.text,
            cand_start=best_start,
            cand_end=best_end,
            cand_text=best_text,
            descriptor_terms=descriptor_terms,
        )

        for ev in span_constraint.evidence:
            ev_start = int(ev.start)
            ev_end = int(ev.end)
            if not is_valid_offsets(text, ev_start, ev_end) or ev_start >= ev_end:
                continue
            if not _span_offsets_overlap(m.span.start, m.span.end, ev_start, ev_end):
                continue
            ev_text = text[ev_start:ev_end]
            if not _boundary_candidate_compatible(m.span.text, ev_text):
                continue
            score = _boundary_alignment_score(
                mention_start=m.span.start,
                mention_end=m.span.end,
                mention_text=m.span.text,
                cand_start=ev_start,
                cand_end=ev_end,
                cand_text=ev_text,
                descriptor_terms=descriptor_terms,
            )
            if score > best_score:
                best_score = score
                best_start, best_end, best_text = ev_start, ev_end, ev_text

        if best_start == m.span.start and best_end == m.span.end:
            out.append(m)
            continue
        if (best_score - _boundary_alignment_score(
            mention_start=m.span.start,
            mention_end=m.span.end,
            mention_text=m.span.text,
            cand_start=m.span.start,
            cand_end=m.span.end,
            cand_text=m.span.text,
            descriptor_terms=descriptor_terms,
        )) < float(min_score_gain):
            out.append(m)
            continue

        calibrated = Mention(
            span_id=f"{m.span_id}__ecal",
            span=Span(
                text=best_text,
                start=best_start,
                end=best_end,
                provenance={
                    "op": "EXPERT_EVIDENCE_BOUNDARY_CALIBRATION",
                    "source_span_id": m.span_id,
                },
            ),
            ent_type=m.ent_type,
            confidence=m.confidence,
            evidence=_dedupe_evidence(
                _merge_evidence(
                    m.evidence,
                    [Evidence(quote=best_text, start=best_start, end=best_end)],
                )
            ),
            rationale=f"{m.rationale}|pipeline:expert_evidence_boundary_calibration",
        )
        out.append(calibrated)
        calibrated_ids.append(calibrated.span_id)

    out, _ = _dedupe_mentions_by_offset_and_type(out)
    out = sorted(out, key=lambda x: (x.span.start, x.span.end, x.span_id))
    return out, {
        "enabled": True,
        "calibrated_count": len(calibrated_ids),
        "calibrated_span_ids": calibrated_ids,
    }


def _is_candidate_recall_entity_like(span_text: str) -> bool:
    text = span_text.strip()
    if not text or len(text) > 140:
        return False
    tokens = re.findall(r"[A-Za-z0-9]+", text)
    if not tokens or len(tokens) > 8:
        return False

    has_digit = any(any(ch.isdigit() for ch in t) for t in tokens)
    has_upper = any(any(ch.isupper() for ch in t) for t in tokens)
    has_symbol = ("-" in text) or ("_" in text) or has_digit
    if has_symbol:
        return not bool(re.fullmatch(r"[IVXLCM]+", text.strip()))
    if has_upper:
        return True
    # Accept multi-token phrases as potentially entity-like, reject very short lowercase singletons.
    if len(tokens) >= 2:
        return True
    return len(tokens[0]) >= 4


def _compile_lexical_rules(
    rules: list[Any],
    default_ent_type: str,
) -> dict[str, dict[str, Any]]:
    compiled: dict[str, dict[str, Any]] = {}
    for item in rules:
        term = ""
        ent_type = default_ent_type
        confidence = 0.9

        if isinstance(item, str):
            term = item.strip()
        elif isinstance(item, dict):
            term = str(item.get("term", item.get("text", ""))).strip()
            ent_type = str(item.get("ent_type", default_ent_type)).strip() or default_ent_type
            confidence = float(item.get("confidence", 0.9))
        else:
            continue

        if not term:
            continue
        compiled[term.lower()] = {
            "ent_type": ent_type,
            "confidence": max(0.0, min(1.0, confidence)),
        }
    return compiled


def _accumulate_cost(total: UsageCost, delta: UsageCost) -> None:
    total.calls += delta.calls
    total.prompt_tokens = _sum_opt(total.prompt_tokens, delta.prompt_tokens)
    total.completion_tokens = _sum_opt(total.completion_tokens, delta.completion_tokens)
    total.total_tokens = _sum_opt(total.total_tokens, delta.total_tokens)
    total.latency_ms.extend(delta.latency_ms)
    total.debate_turns += delta.debate_turns
    total.debate_triggered += delta.debate_triggered


def _sum_opt(a: int | None, b: int | None) -> int | None:
    if a is None and b is None:
        return None
    return int(a or 0) + int(b or 0)


def _trim_whitespace_bounds(text: str, start: int, end: int) -> tuple[int, int] | None:
    s = start
    e = end
    n = len(text)
    while s < e and s < n and text[s].isspace():
        s += 1
    while e > s and e - 1 >= 0 and text[e - 1].isspace():
        e -= 1
    if not is_valid_offsets(text, s, e) or s >= e:
        return None
    return s, e


def _is_short_symbol(token: str) -> bool:
    t = token.strip()
    if not t or " " in t:
        return False
    if len(t) > 8:
        return False
    has_alpha = any(ch.isalpha() for ch in t)
    has_digit = any(ch.isdigit() for ch in t)
    return has_alpha and has_digit


def _normalize_terms(raw: Any) -> list[str]:
    items: list[Any]
    if raw is None:
        return []
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, (list, tuple, set)):
        items = list(raw)
    else:
        return []

    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        term = str(item).strip().lower()
        if not term or term in seen:
            continue
        seen.add(term)
        out.append(term)
    return out


def _schema_descriptor_terms(schema: Any) -> list[str]:
    # Keep schema term extraction keyword-free in framework code.
    terms: set[str] = set()
    name_terms: set[str] = set()
    desc_counts: Counter[str] = Counter()
    for et in getattr(schema, "entity_types", []) or []:
        name = str(getattr(et, "name", "") or "")
        description = str(getattr(et, "description", "") or "")

        for tok in re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}", name.lower()):
            if len(tok) > 24:
                continue
            name_terms.add(tok)

        for tok in re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}", description.lower()):
            if len(tok) > 24:
                continue
            desc_counts[tok] += 1

    terms.update(name_terms)
    for tok, count in desc_counts.items():
        if (
            tok in name_terms
            or count >= 2
            or any(ch.isdigit() for ch in tok)
            or "-" in tok
        ):
            terms.add(tok)
    return sorted(terms)


def _schema_exclusion_terms(schema: Any) -> list[str]:
    # No keyword-based exclusion extraction in framework code.
    _ = schema
    return []


def _augment_candidate_set(
    candidate_set: CandidateSet,
    text: str,
    proposals: list[dict[str, Any]],
    max_new_spans: int = 20,
) -> tuple[list[str], dict[str, Any]]:
    by_offsets = {(span.start, span.end): sid for sid, span in candidate_set.spans.items()}
    added_span_ids: list[str] = []
    used_proposals = 0

    next_idx = _next_aug_index(candidate_set.spans.keys())
    for proposal in proposals:
        if len(added_span_ids) >= max_new_spans:
            break

        start = int(proposal.get("start", -1))
        end = int(proposal.get("end", -1))
        trimmed = _trim_whitespace_bounds(text=text, start=start, end=end)
        if trimmed is None:
            continue
        start, end = trimmed
        if not is_valid_offsets(text, start, end) or start >= end:
            continue

        key = (start, end)
        if key in by_offsets:
            continue

        span_id = f"aug_{next_idx:04d}"
        while span_id in candidate_set.spans:
            next_idx += 1
            span_id = f"aug_{next_idx:04d}"
        next_idx += 1

        candidate_set.spans[span_id] = Span(
            text=text[start:end],
            start=start,
            end=end,
            provenance={
                "source_agent": str(proposal.get("source", "")),
                "op": "AUGMENT",
                "rationale": str(proposal.get("rationale", "")),
                "confidence": float(proposal.get("confidence", 0.0)),
                "evidence": proposal.get("evidence", []),
            },
        )
        by_offsets[key] = span_id
        added_span_ids.append(span_id)
        used_proposals += 1

    candidate_set.has_entity = bool(candidate_set.spans)
    trace = {
        "enabled": True,
        "proposals_total": len(proposals),
        "proposals_used": used_proposals,
        "added_count": len(added_span_ids),
        "added_span_ids": added_span_ids,
    }
    return added_span_ids, trace


def _next_aug_index(span_ids: Any) -> int:
    max_idx = 0
    for sid in span_ids:
        sid_s = str(sid)
        if sid_s.startswith("aug_"):
            tail = sid_s.removeprefix("aug_")
            if tail.isdigit():
                max_idx = max(max_idx, int(tail))
    return max_idx + 1


def _filter_span_proposals(
    proposals: list[dict[str, Any]],
    valid_types: set[str],
    min_confidence: float,
    require_schema_type_hint: bool,
    reject_negative_rationale: bool,
    require_evidence: bool = False,
    require_evidence_anchor: bool = False,
    max_tokens: int = 0,
    entity_like_only: bool = False,
    source_min_confidence: dict[str, float] | None = None,
    reject_overlap_sources: set[str] | None = None,
    existing_spans: list[tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    source_min_confidence = source_min_confidence or {}
    reject_overlap_sources = reject_overlap_sources or set()
    existing_spans = existing_spans or []
    for p in proposals:
        source = str(p.get("source", "")).strip().lower()
        conf = float(p.get("confidence", 0.0))
        effective_min_confidence = max(
            float(min_confidence),
            float(source_min_confidence.get(source, min_confidence)),
        )
        if conf < effective_min_confidence:
            continue
        if reject_negative_rationale and _has_negative_entity_rationale(
            str(p.get("rationale", ""))
        ):
            continue
        span_text = str(p.get("text", ""))
        start = int(p.get("start", -1))
        end = int(p.get("end", -1))
        if max_tokens > 0 and len([t for t in span_text.split() if t]) > max_tokens:
            continue
        if (
            source in reject_overlap_sources
            and start >= 0
            and end > start
            and any(_span_offsets_overlap(start, end, ex_start, ex_end) for ex_start, ex_end in existing_spans)
        ):
            continue
        evidence_items = p.get("evidence", []) or []
        if require_evidence and not evidence_items:
            continue
        if require_evidence_anchor and not _proposal_has_evidence_anchor(p):
            continue
        if require_schema_type_hint:
            hints = [str(x) for x in (p.get("type_hints", []) or []) if isinstance(x, str)]
            if not any(h in valid_types for h in hints):
                continue
        if entity_like_only and not _looks_entity_like_proposal(
            span_text=span_text,
            rationale=str(p.get("rationale", "")),
        ):
            continue
        out.append(p)
    return out


def _has_negative_entity_rationale(rationale: str) -> bool:
    # Keep this guard keyword-free in framework code.
    _ = rationale
    return False


def _proposal_has_evidence_anchor(proposal: dict[str, Any]) -> bool:
    start = int(proposal.get("start", -1))
    end = int(proposal.get("end", -1))
    span_text = str(proposal.get("text", ""))
    if start < 0 or end <= start:
        return False
    for ev in proposal.get("evidence", []) or []:
        if not isinstance(ev, dict):
            continue
        ev_quote = str(ev.get("quote", ""))
        ev_start = int(ev.get("start", -1))
        ev_end = int(ev.get("end", -1))
        if ev_start < 0 or ev_end <= ev_start:
            continue
        if ev_start <= start and end <= ev_end:
            return True
        if start <= ev_start and ev_end <= end:
            return True
        if span_text and span_text in ev_quote:
            return True
    return False


def _looks_entity_like_proposal(span_text: str, rationale: str) -> bool:
    stripped = span_text.strip()
    if not stripped:
        return False
    if len(stripped) > 160:
        return False

    tokens = re.findall(r"[A-Za-z0-9]+", stripped)
    if not tokens:
        return False
    if len(tokens) > 12:
        return False

    if all(tok.isdigit() for tok in tokens):
        return False
    if all(re.fullmatch(r"[ivxlcdm]+", tok.lower()) for tok in tokens):
        return False

    has_digit = any(any(ch.isdigit() for ch in t) for t in tokens)
    has_caps = any(any(ch.isupper() for ch in t) for t in tokens)
    has_hyphen = "-" in stripped or "_" in stripped
    multi_token = len(tokens) >= 2
    if has_digit or has_caps or has_hyphen or multi_token:
        return True

    rationale_tokens = re.findall(r"[A-Za-z0-9]+", rationale)
    if len(rationale_tokens) >= 2:
        return True

    if len(tokens) == 1 and len(tokens[0]) <= 1:
        return False
    return any(len(tok) >= 3 for tok in tokens)


def _drop_mentions_with_negative_rationale(
    mentions: list[Mention],
) -> tuple[list[Mention], int]:
    kept: list[Mention] = []
    dropped = 0
    for mention in mentions:
        if _has_negative_entity_rationale(mention.rationale):
            dropped += 1
            continue
        kept.append(mention)
    return kept, dropped


def _normalize_candidate_span_boundaries(
    candidate_set: CandidateSet,
    text: str,
    left_modifiers: list[str],
    right_modifiers: list[str],
    enable_hyphen_left: bool,
    max_right_expansion_steps: int = 1,
) -> tuple[list[str], dict[str, Any]]:
    left_modifiers_l = {x.lower() for x in left_modifiers}
    right_modifiers_l = {x.lower() for x in right_modifiers}
    updated_span_ids: list[str] = []
    for span_id, span in candidate_set.spans.items():
        new_start, new_end = _expand_span_boundaries(
            text=text,
            start=span.start,
            end=span.end,
            left_modifiers=left_modifiers_l,
            right_modifiers=right_modifiers_l,
            enable_hyphen_left=enable_hyphen_left,
            max_right_expansion_steps=max_right_expansion_steps,
        )
        if new_start == span.start and new_end == span.end:
            continue
        if not is_valid_offsets(text, new_start, new_end) or new_start >= new_end:
            continue
        span.start = new_start
        span.end = new_end
        span.text = text[new_start:new_end]
        updated_span_ids.append(span_id)

    return updated_span_ids, {
        "enabled": True,
        "updated_span_ids": updated_span_ids,
        "updated_count": len(updated_span_ids),
    }


def _expand_span_boundaries(
    text: str,
    start: int,
    end: int,
    left_modifiers: set[str],
    right_modifiers: set[str],
    enable_hyphen_left: bool,
    max_right_expansion_steps: int = 1,
) -> tuple[int, int]:
    new_start = start
    new_end = end

    left_word = _word_before(text, new_start)
    if left_word is not None:
        token, token_start, _ = left_word
        if token.lower() in left_modifiers:
            new_start = token_start

    # Support multi-step suffix expansion for modifier continuations.
    steps = max(1, int(max_right_expansion_steps))
    for _ in range(steps):
        right_word = _word_after(text, new_end)
        if right_word is None:
            break
        token, _, token_end = right_word
        if token.lower() not in right_modifiers:
            break
        new_end = token_end

    if enable_hyphen_left:
        hyphen_left = _hyphen_left_token(text, new_start)
        if hyphen_left is not None:
            new_start = hyphen_left

    return new_start, new_end


def _summarize_rag_expert_alignment(
    rag_handoff: dict[str, Any],
    constraints: ExpertConstraints,
) -> dict[str, Any]:
    hinted = rag_handoff.get("per_span_hints", {}) or {}
    if not isinstance(hinted, dict):
        hinted = {}

    hinted_span_ids = set(hinted.keys())
    constrained_span_ids = set(constraints.per_span.keys())
    covered = sorted(hinted_span_ids.intersection(constrained_span_ids))
    missed = sorted(hinted_span_ids.difference(constrained_span_ids))

    return {
        "handoff_id": str(rag_handoff.get("handoff_id", "")),
        "hinted_spans": len(hinted_span_ids),
        "covered_spans": len(covered),
        "missed_spans": len(missed),
        "coverage": (len(covered) / len(hinted_span_ids)) if hinted_span_ids else 1.0,
        "missed_span_ids": missed,
    }


def _word_before(text: str, idx: int) -> tuple[str, int, int] | None:
    i = idx - 1
    while i >= 0 and text[i].isspace():
        i -= 1
    if i < 0 or not text[i].isalnum():
        return None
    end = i + 1
    while i >= 0 and (text[i].isalnum() or text[i] in {"_", "-"}):
        i -= 1
    start = i + 1
    return text[start:end], start, end


def _word_after(text: str, idx: int) -> tuple[str, int, int] | None:
    i = idx
    n = len(text)
    while i < n and text[i].isspace():
        i += 1
    if i >= n or not text[i].isalnum():
        return None
    start = i
    while i < n and (text[i].isalnum() or text[i] in {"_", "-"}):
        i += 1
    end = i
    return text[start:end], start, end


def _hyphen_left_token(text: str, start: int) -> int | None:
    i = start - 1
    while i >= 0 and text[i].isspace():
        i -= 1
    if i < 0 or text[i] != "-":
        return None
    j = i - 1
    while j >= 0 and text[j].isspace():
        j -= 1
    if j < 0 or not text[j].isalnum():
        return None
    end = j + 1
    while j >= 0 and (text[j].isalnum() or text[j] in {"_", "-"}):
        j -= 1
    tok_start = j + 1
    token = text[tok_start:end]
    if len(token) > 20:
        return None
    return tok_start

