from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


class NullRetriever:
    def retrieve(self, query_plan: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return [], {"enabled": False, "source": "null", "queries_attempted": 0, "docs_returned": 0}


class WikipediaRetriever:
    def __init__(self, settings: dict[str, Any] | None = None):
        settings = dict(settings or {})
        wiki_cfg = settings.get("wikipedia", settings)
        self.enabled = bool(wiki_cfg.get("enabled", True))
        self.language = str(wiki_cfg.get("language", "en")).strip() or "en"
        self.max_requests_per_sample = int(wiki_cfg.get("max_requests_per_sample", 6))
        self.max_results_per_request = int(wiki_cfg.get("max_results_per_request", 2))
        self.summary_chars = int(wiki_cfg.get("summary_chars", 420))
        self.timeout_s = int(wiki_cfg.get("timeout_s", 12))
        self.max_retries = int(wiki_cfg.get("max_retries", 1))
        self.retry_backoff_s = float(wiki_cfg.get("retry_backoff_s", 0.75))
        self.user_agent = str(
            wiki_cfg.get(
                "user_agent",
                "multi-agent-ner/1.0 (https://github.com/openai; contact=local-runtime)",
            )
        )
        self.api_url = f"https://{self.language}.wikipedia.org/w/api.php"

    def retrieve(self, query_plan: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not self.enabled:
            return [], {"enabled": False, "source": "wikipedia", "queries_attempted": 0, "docs_returned": 0}

        queries = [
            item
            for item in (query_plan.get("candidate_queries", []) or [])
            if isinstance(item, dict) and str(item.get("query", "")).strip()
        ][: max(0, self.max_requests_per_sample)]

        documents: list[dict[str, Any]] = []
        errors: list[str] = []
        queries_attempted = 0
        for q_idx, query in enumerate(queries, start=1):
            queries_attempted += 1
            try:
                search_hits = self._search(str(query.get("query", "")))
                if not search_hits:
                    continue
                titles = [str(hit.get("title", "")).strip() for hit in search_hits[: self.max_results_per_request]]
                titles = [title for title in titles if title]
                if not titles:
                    continue
                pages = self._page_summaries(titles)
                for rank, hit in enumerate(search_hits[: self.max_results_per_request], start=1):
                    title = str(hit.get("title", "")).strip()
                    page = pages.get(title, {})
                    summary = str(page.get("extract", "")).strip()
                    if self.summary_chars > 0:
                        summary = summary[: self.summary_chars].strip()
                    if not summary:
                        continue
                    documents.append(
                        {
                            "doc_id": f"wiki_{q_idx:02d}_{rank:02d}",
                            "query_id": str(query.get("q_id", "")),
                            "query": str(query.get("query", "")),
                            "source": "wikipedia",
                            "title": title,
                            "url": str(page.get("fullurl", "")),
                            "summary": summary,
                            "linked_span_ids": [
                                str(sid)
                                for sid in query.get("span_ids", []) or []
                                if str(sid).strip()
                            ],
                            "relevance": self._relevance_score(hit=hit, rank=rank),
                        }
                    )
            except Exception as exc:
                errors.append(f"{query.get('q_id', 'query')}:{exc}")

        trace = {
            "enabled": True,
            "source": "wikipedia",
            "queries_attempted": queries_attempted,
            "docs_returned": len(documents),
            "errors": errors,
        }
        return documents, trace

    def _search(self, query: str) -> list[dict[str, Any]]:
        payload = self._request_json(
            {
                "action": "query",
                "list": "search",
                "format": "json",
                "utf8": "1",
                "srlimit": str(max(1, self.max_results_per_request)),
                "srsearch": query,
            }
        )
        search_results = payload.get("query", {}).get("search", [])
        if not isinstance(search_results, list):
            return []
        hits: list[dict[str, Any]] = []
        for item in search_results:
            if not isinstance(item, dict):
                continue
            hits.append(
                {
                    "title": str(item.get("title", "")),
                    "pageid": item.get("pageid"),
                    "size": int(item.get("size", 0) or 0),
                    "wordcount": int(item.get("wordcount", 0) or 0),
                }
            )
        return hits

    def _page_summaries(self, titles: list[str]) -> dict[str, dict[str, Any]]:
        payload = self._request_json(
            {
                "action": "query",
                "prop": "extracts|info",
                "inprop": "url",
                "exintro": "1",
                "explaintext": "1",
                "redirects": "1",
                "format": "json",
                "titles": "|".join(titles),
            }
        )
        pages = payload.get("query", {}).get("pages", {})
        if not isinstance(pages, dict):
            return {}
        title_map: dict[str, dict[str, Any]] = {}
        for item in pages.values():
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            if not title:
                continue
            title_map[title] = {
                "extract": str(item.get("extract", "")),
                "fullurl": str(item.get("fullurl", "")),
            }
        return title_map

    def _request_json(self, params: dict[str, str]) -> dict[str, Any]:
        qs = urllib.parse.urlencode(params)
        url = f"{self.api_url}?{qs}"
        attempts = max(0, self.max_retries) + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            req = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": self.user_agent,
                },
                method="GET",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                last_error = exc
                if attempt >= attempts - 1:
                    break
                time.sleep(self.retry_backoff_s * (2**attempt))
        assert last_error is not None
        raise RuntimeError(f"Wikipedia retrieval failed: {last_error}") from last_error

    def _relevance_score(self, *, hit: dict[str, Any], rank: int) -> float:
        base = max(0.0, 1.0 - 0.15 * (rank - 1))
        size_bonus = min(0.15, float(hit.get("wordcount", 0)) / 5000.0)
        return round(min(0.99, base + size_bonus), 4)
