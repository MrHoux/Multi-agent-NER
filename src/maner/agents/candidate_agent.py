from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any

from maner.core.alignment import align_substring_offsets
from maner.core.prompting import PromptManager
from maner.core.schema import SchemaDefinition
from maner.core.types import CandidateSet, Mention, Span, UsageCost, is_valid_offsets
from maner.llm.client import LLMClient


def _usage_to_cost(usage: Any) -> UsageCost:
    return UsageCost(
        calls=int(getattr(usage, "calls", 0) or 0),
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
        latency_ms=list(getattr(usage, "latency_ms", []) or []),
    )


class CandidateAgent:
    def __init__(
        self,
        llm_client: LLMClient,
        prompt_manager: PromptManager,
        settings: dict[str, Any] | None = None,
    ):
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager
        self.settings = settings or {}

    def run(
        self,
        text: str,
        schema: SchemaDefinition,
        seed_mentions: list[Mention] | None = None,
    ) -> tuple[CandidateSet, UsageCost, dict[str, Any]]:
        seed_mentions = seed_mentions or []
        system, user = self.prompt_manager.render(
            "candidate_agent",
            text=text,
            entity_types=schema.to_prompt_block(),
            seed_mentions=[
                {
                    "span_id": mention.span_id,
                    "text": mention.span.text,
                    "start": mention.span.start,
                    "end": mention.span.end,
                    "ent_type": mention.ent_type,
                    "confidence": mention.confidence,
                }
                for mention in seed_mentions
            ],
        )

        context = None
        if self.llm_client.provider == "mock":
            context = {"mock_result": self._heuristic_candidates(text)}

        llm_result = self.llm_client.chat_json(
            system_prompt=system,
            user_prompt=user,
            task="candidate_agent",
            context=context,
        )
        payload = llm_result.parsed_json
        if payload.get("mock") and "result" in payload:
            payload = payload["result"]

        candidate_set = self._parse_candidate_payload(payload, text)
        fallback_added = self._inject_generic_fallback_spans(candidate_set, text, schema)
        cost = _usage_to_cost(llm_result.usage)
        trace = {
            "agent": "candidate",
            "raw": llm_result.content,
            "parsed": payload,
            "num_spans": len(candidate_set.spans),
            "seed_mentions": len(seed_mentions),
            "fallback_added": fallback_added,
            "fallback_mode": "generic",
        }
        return candidate_set, cost, trace

    def _heuristic_candidates(self, text: str) -> dict[str, Any]:
        spans: list[dict[str, Any]] = []
        seen: set[tuple[int, int]] = set()

        patterns = [
            r"\b[A-Z][A-Za-z0-9\-]*(?:\s+[A-Z][A-Za-z0-9\-]*)*\b",
            r"\b[A-Z]{2,}\b",
            r"\b\d{4}\b",
        ]
        for pattern in patterns:
            for m in re.finditer(pattern, text):
                start, end = m.span()
                if (start, end) in seen:
                    continue
                seen.add((start, end))
                spans.append({"start": start, "end": end, "text": text[start:end]})

        stop_words = {"the", "and", "for", "with", "from", "that", "this", "into"}
        for m in re.finditer(r"\b[A-Za-z][A-Za-z0-9\-]{2,}\b", text):
            token = m.group(0)
            if token.lower() in stop_words:
                continue
            start, end = m.span()
            if (start, end) in seen:
                continue
            seen.add((start, end))
            spans.append({"start": start, "end": end, "text": token})

        spans = sorted(spans, key=lambda x: (x["start"], x["end"]))
        labeled = []
        for idx, span in enumerate(spans, start=1):
            labeled.append({"span_id": f"sp_{idx:04d}", **span})
        return {"has_entity": len(labeled) > 0, "spans": labeled}

    def _parse_candidate_payload(self, payload: dict[str, Any], text: str) -> CandidateSet:
        raw_spans = payload.get("spans", [])
        spans: dict[str, Span] = {}
        seen_entities: set[tuple[int, int, str]] = set()
        next_idx = _next_span_index(spans.keys())

        if isinstance(raw_spans, dict):
            iterator = [dict({"span_id": k}, **v) for k, v in raw_spans.items()]
        else:
            iterator = list(raw_spans)

        for item in iterator:
            raw_span_id = str(item.get("span_id") or "").strip()
            start = int(item.get("start", -1))
            end = int(item.get("end", -1))
            quote = str(item.get("text", ""))

            if quote:
                aligned = align_substring_offsets(
                    text=text,
                    quote=quote,
                    start_hint=start,
                    end_hint=end,
                )
                if aligned is not None:
                    start, end = aligned
                    quote = text[start:end]
                elif is_valid_offsets(text, start, end) and start != end:
                    quote = text[start:end]
                else:
                    continue
            elif is_valid_offsets(text, start, end) and start != end:
                quote = text[start:end]
            else:
                continue

            if not is_valid_offsets(text, start, end) or start == end:
                continue
            trimmed = _trim_whitespace_bounds(text=text, start=start, end=end)
            if trimmed is None:
                continue
            start, end = trimmed
            quote = text[start:end]

            entity_key = (start, end, quote)
            if entity_key in seen_entities:
                continue

            span_id = raw_span_id or f"sp_{next_idx:04d}"
            if raw_span_id and span_id in spans:
                existing = spans[span_id]
                if existing.start == start and existing.end == end and existing.text == quote:
                    continue
                span_id = f"sp_{next_idx:04d}"
            while span_id in spans:
                next_idx += 1
                span_id = f"sp_{next_idx:04d}"
            spans[span_id] = Span(text=quote, start=start, end=end)
            seen_entities.add(entity_key)
            parsed_idx = _parse_span_index(span_id)
            if parsed_idx is not None:
                next_idx = max(next_idx, parsed_idx + 1)

        return CandidateSet(has_entity=bool(payload.get("has_entity", bool(spans))), spans=spans)

    @staticmethod
    def serialize(candidate_set: CandidateSet) -> dict[str, Any]:
        return asdict(candidate_set)

    def _inject_generic_fallback_spans(
        self,
        candidate_set: CandidateSet,
        text: str,
        schema: SchemaDefinition,
    ) -> int:
        enable = bool(self.settings.get("enable_generic_fallback", True))
        only_when_empty = bool(self.settings.get("fallback_only_when_empty", True))
        max_added = max(0, int(self.settings.get("fallback_max_added", 32)))
        min_token_len = max(1, int(self.settings.get("fallback_min_token_len", 2)))
        max_total_spans = max(1, int(self.settings.get("fallback_max_total_spans", 120)))

        if not enable:
            return 0
        if only_when_empty and candidate_set.spans:
            return 0
        if len(candidate_set.spans) >= max_total_spans:
            return 0

        by_offsets = {(span.start, span.end) for span in candidate_set.spans.values()}
        next_idx = _next_span_index(candidate_set.spans.keys())
        added = 0

        for start, end in self._generic_fallback_offsets(
            text,
            schema,
            min_token_len=min_token_len,
        ):
            if added >= max_added:
                break
            if len(candidate_set.spans) >= max_total_spans:
                break
            if (start, end) in by_offsets:
                continue
            span_id = f"sp_{next_idx:04d}"
            while span_id in candidate_set.spans:
                next_idx += 1
                span_id = f"sp_{next_idx:04d}"
            next_idx += 1
            candidate_set.spans[span_id] = Span(text=text[start:end], start=start, end=end)
            by_offsets.add((start, end))
            added += 1

        if added > 0:
            candidate_set.has_entity = True
        return added

    def _generic_fallback_offsets(
        self,
        text: str,
        schema: SchemaDefinition,
        min_token_len: int = 2,
    ) -> list[tuple[int, int]]:
        offsets: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        stop_tokens = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "for",
            "with",
            "from",
            "that",
            "this",
            "these",
            "those",
            "into",
            "onto",
            "about",
            "above",
            "below",
            "under",
            "over",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
        }
        lower_only_stop_tokens = {
            "study",
            "studies",
            "result",
            "results",
            "method",
            "methods",
            "cell",
            "cells",
            "expression",
            "complex",
            "complexes",
            "extract",
            "extracts",
            "vector",
            "vectors",
            "using",
            "with",
            "from",
            "into",
            "that",
            "this",
            "these",
            "those",
        }
        common_non_entity_acronyms = {
            "CPU",
            "GPU",
            "RAM",
            "HTTP",
            "HTTPS",
            "JSON",
            "SQL",
            "API",
            "URL",
            "USA",
            "EU",
            "UK",
        }
        enable_parenthetical_alias = bool(
            self.settings.get("fallback_enable_parenthetical_alias", True)
        )
        enable_descriptor_phrases = bool(
            self.settings.get("fallback_enable_descriptor_phrases", True)
        )
        enable_list_context = bool(self.settings.get("fallback_enable_list_context", True))
        enable_symbolic_patterns = bool(
            self.settings.get(
                "fallback_enable_symbolic_patterns",
                self.settings.get("fallback_enable_schema_profile", True),
            )
        )
        max_descriptor_phrase_tokens = max(
            2,
            int(self.settings.get("fallback_descriptor_phrase_max_tokens", 6)),
        )

        patterns = [
            (
                # Proper-noun style multiword mentions.
                r"\b[A-Z][A-Za-z0-9\-]{1,}(?:\s+[A-Z][A-Za-z0-9\-]{1,})*\b",
                0,
                "title_like",
            ),
            (
                # Symbolic identifiers containing digits.
                r"\b[A-Za-z][A-Za-z0-9\-_]{0,20}\d+[A-Za-z0-9\-_]{0,20}\b",
                re.IGNORECASE,
                "generic",
            ),
            (
                # Compact all-caps abbreviations.
                r"\b[A-Z]{2,8}\b",
                0,
                "acronym",
            ),
        ]
        for pattern, flags, kind in patterns:
            for m in re.finditer(pattern, text, flags=flags):
                start, end = m.span()
                token = text[start:end]
                token_norm = token.strip()
                if len(token_norm) < min_token_len:
                    continue
                if token_norm.lower() in stop_tokens:
                    continue
                if token_norm.isdigit():
                    continue
                if kind == "acronym" and token_norm.upper() in common_non_entity_acronyms:
                    continue
                if (start, end) in seen:
                    continue
                seen.add((start, end))
                offsets.append((start, end))

        if enable_symbolic_patterns:
            symbolic_patterns = [
                # Symbol chains with explicit separators, e.g. "A - 12" / "X / Y".
                (
                    r"\b[A-Za-z][A-Za-z0-9]{0,12}\s*(?:\s*[-/]\s*[A-Za-z0-9]{1,12}){1,4}(?:\s*\+)?",
                    0,
                    False,
                ),
                # Plus-suffixed marker forms, e.g. "CD4 +".
                (
                    r"\b[A-Za-z][A-Za-z0-9]{0,12}\s*\+",
                    0,
                    False,
                ),
                # Dot-separated compact aliases, e.g. "U . S".
                (
                    r"\b[A-Za-z]{1,6}\s*\.\s*[A-Za-z0-9]{1,8}",
                    0,
                    False,
                ),
                # Mixed-case compact symbols, e.g. "Sp1" / "X12".
                (r"\b[A-Za-z][A-Za-z0-9\-_]{1,12}\b", 0, True),
                # Prefix-linked aliases, e.g. "pre-X1" / "pro-ABC".
                (
                    r"\b(?:pre|pro)\s*-\s*[A-Za-z][A-Za-z0-9\-_]{1,24}\b",
                    re.IGNORECASE,
                    False,
                ),
            ]
            for pattern, flags, strict_compact_filter in symbolic_patterns:
                for m in re.finditer(pattern, text, flags=flags):
                    start, end = m.span()
                    token = text[start:end].strip()
                    if len(token) < min_token_len:
                        continue
                    if token.lower() in stop_tokens:
                        continue
                    if (start, end) in seen:
                        continue
                    # For compact symbols, require stronger signal to avoid broad noise.
                    if strict_compact_filter and re.fullmatch(
                        r"[A-Za-z][A-Za-z0-9\-_]{1,12}",
                        token,
                    ):
                        upper_count = sum(ch.isupper() for ch in token)
                        has_digit = any(ch.isdigit() for ch in token)
                        if upper_count < 2 and not has_digit:
                            continue
                    seen.add((start, end))
                    offsets.append((start, end))

        if enable_parenthetical_alias:
            for m in re.finditer(r"\(\s*([A-Za-z][A-Za-z0-9\-_]{1,24})\s*\)", text):
                start, end = m.span(1)
                token = text[start:end]
                if token.lower() in stop_tokens:
                    continue
                if len(token) < min_token_len:
                    continue
                if (start, end) in seen:
                    continue
                seen.add((start, end))
                offsets.append((start, end))

        if enable_descriptor_phrases:
            descriptor_terms = _collect_schema_descriptor_terms(
                schema,
                extra_terms=self.settings.get("fallback_descriptor_terms", []),
            )
            for start, end in _descriptor_phrase_offsets(
                text=text,
                descriptor_terms=descriptor_terms,
                max_tokens=max_descriptor_phrase_tokens,
            ):
                phrase = text[start:end].strip()
                if len(phrase) < min_token_len:
                    continue
                if phrase.lower() in stop_tokens:
                    continue
                if (start, end) in seen:
                    continue
                seen.add((start, end))
                offsets.append((start, end))

        if enable_list_context:
            for m in re.finditer(r"\b[A-Za-z][A-Za-z0-9\-_]{2,24}\b", text):
                start, end = m.span()
                token = text[start:end]
                token_l = token.lower()
                if token_l in stop_tokens or token_l in lower_only_stop_tokens:
                    continue
                if any(ch.isupper() for ch in token) or any(ch.isdigit() for ch in token):
                    continue
                # Focus on coordinated/enumerated contexts to avoid broad noise.
                left = text[max(0, start - 12) : start]
                right = text[end : min(len(text), end + 16)]
                near_sep = (
                    ("," in left)
                    or ("," in right)
                    or ("/" in left)
                    or ("/" in right)
                    or bool(re.search(r"\b(and|or)\b", left + right, flags=re.IGNORECASE))
                )
                if not near_sep:
                    continue

                window = text[max(0, start - 48) : min(len(text), end + 48)]
                # Require at least one stronger item nearby (caps or digit token).
                if not re.search(r"\b(?:[A-Z][A-Za-z0-9\-_]*|[A-Za-z0-9\-_]*\d+[A-Za-z0-9\-_]*)\b", window):
                    continue
                if (start, end) in seen:
                    continue
                seen.add((start, end))
                offsets.append((start, end))

        offsets.sort(key=lambda x: (x[0], x[1]))
        return offsets


def _collect_schema_descriptor_terms(
    schema: SchemaDefinition,
    extra_terms: Any = None,
) -> list[str]:
    stop = {
        "named",
        "mention",
        "mentions",
        "text",
        "span",
        "spans",
        "entity",
        "entities",
        "including",
        "exclude",
        "explicitly",
        "specific",
        "minimal",
        "exact",
        "covering",
        "only",
        "used",
        "when",
        "unless",
        "such",
        "as",
        "and",
        "or",
    }
    generic_noise = {
        "including",
        "include",
        "covers",
        "covering",
        "example",
        "examples",
        "general",
    }
    exclusion_markers = {
        "exclude",
        "excluding",
        "non-entity",
        "not ",
        "unless ",
        "without ",
    }
    terms: set[str] = set()
    for et in schema.entity_types:
        name = str(et.name or "").lower()
        description = str(et.description or "")
        for tok in re.findall(r"[a-z][a-z0-9\-_]{2,24}", name):
            if tok in stop or tok in generic_noise:
                continue
            terms.add(tok)

        clauses = re.split(r"[.;]", description)
        for clause in clauses:
            clause_l = clause.lower()
            if any(marker in clause_l for marker in exclusion_markers):
                continue
            for tok in re.findall(r"[a-z][a-z0-9\-_]{2,24}", clause_l):
                if tok in stop or tok in generic_noise:
                    continue
                terms.add(tok)

    if isinstance(extra_terms, list):
        for t in extra_terms:
            tok = str(t).strip().lower()
            if tok and tok not in stop:
                terms.add(tok)
    elif isinstance(extra_terms, str):
        tok = extra_terms.strip().lower()
        if tok and tok not in stop:
            terms.add(tok)

    return sorted(terms)


def _descriptor_phrase_offsets(
    text: str,
    descriptor_terms: list[str],
    max_tokens: int,
) -> list[tuple[int, int]]:
    if not descriptor_terms:
        return []

    escaped_terms = sorted({re.escape(t) for t in descriptor_terms if t}, key=len, reverse=True)
    if not escaped_terms:
        return []

    term_pattern = re.compile(r"\b(?:" + "|".join(escaped_terms) + r")\b", flags=re.IGNORECASE)
    offsets: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "for",
        "with",
        "from",
        "in",
        "on",
        "at",
        "by",
        "into",
        "during",
        "between",
        "after",
        "before",
        "while",
        "when",
        "where",
        "which",
        "that",
        "this",
        "these",
        "those",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "as",
        "if",
        "than",
        "then",
        "such",
        "single",
        "novel",
        "major",
        "minor",
        "primary",
        "secondary",
        "transient",
        "increasing",
        "decreasing",
        "established",
        "previous",
        "double",
        "single",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "containing",
    }
    bridge_words = {"-", "/", "(", ")", "[", "]"}

    for m in term_pattern.finditer(text):
        start = m.start()
        end = m.end()
        cursor = start
        token_count = len(re.findall(r"[A-Za-z0-9]+", text[start:end]))
        symbol_token_count = 1 if _is_symbol_like_token(text[start:end]) else 0

        while token_count < max_tokens:
            prev = _word_before_token(text, cursor)
            if prev is None:
                break
            tok, tok_start, tok_end = prev
            gap = text[tok_end:cursor]
            if any(ch in ",.;:{}" for ch in gap):
                break
            if gap.strip() and any(ch not in bridge_words and not ch.isspace() for ch in gap):
                break
            lower = tok.lower()
            if lower in stop:
                break
            is_symbol = _is_symbol_like_token(tok)
            if is_symbol and symbol_token_count >= 1:
                break
            if len(tok) > 28:
                break
            cursor = tok_start
            token_count += 1
            if is_symbol:
                symbol_token_count += 1

        phrase = text[cursor:end].strip()
        if not phrase:
            continue
        toks = re.findall(r"[A-Za-z0-9]+", phrase)
        if len(toks) < 2 or len(toks) > max_tokens:
            continue
        if any(tok.lower() in stop for tok in toks):
            continue
        key = (cursor, end)
        if key in seen:
            continue
        seen.add(key)
        offsets.append(key)

    return offsets


def _word_before_token(text: str, idx: int) -> tuple[str, int, int] | None:
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


def _is_symbol_like_token(token: str) -> bool:
    t = token.strip()
    if not t or " " in t:
        return False
    if len(t) > 20:
        return False
    has_alpha = any(ch.isalpha() for ch in t)
    has_digit = any(ch.isdigit() for ch in t)
    if has_alpha and has_digit:
        return True
    return t.isupper() and has_alpha and len(t) <= 8


def _next_span_index(span_ids: Any) -> int:
    max_idx = 0
    for sid in span_ids:
        parsed = _parse_span_index(str(sid))
        if parsed is not None:
            max_idx = max(max_idx, parsed)
    return max_idx + 1


def _parse_span_index(span_id: str) -> int | None:
    if not span_id.startswith("sp_"):
        return None
    tail = span_id[3:]
    if tail.isdigit():
        return int(tail)
    return None
