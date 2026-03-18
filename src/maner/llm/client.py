import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.client import IncompleteRead
from typing import Any

from maner.llm.parsing import JsonParseError, parse_llm_json


@dataclass
class LLMUsage:
    calls: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: list[int] | None = None


@dataclass
class LLMResult:
    content: str
    parsed_json: dict[str, Any]
    usage: LLMUsage


class LLMClient:
    """Unified commercial LLM client with provider adapters."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.provider = config.get("provider", "openai_compatible")
        self.model = config.get("model", "")
        self.base_url = config.get("base_url", "")
        self.api_key = str(config.get("api_key", "") or "")
        self.api_key_env = config.get("api_key_env", "LLM_API_KEY")
        self.timeout_s = int(config.get("timeout_s", 60))
        self.temperature = float(config.get("temperature", 0.0))
        self.max_retries = int(config.get("max_retries", 3))
        self.retry_backoff_s = float(config.get("retry_backoff_s", 1.0))
        self.parse_retries = int(config.get("parse_retries", 1))

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        task: str = "",
        context: dict[str, Any] | None = None,
    ) -> LLMResult:
        total_usage = LLMUsage(calls=0, latency_ms=[])
        parse_attempts = max(0, self.parse_retries) + 1
        last_error: Exception | None = None
        last_content = ""

        for parse_attempt in range(parse_attempts):
            start = time.time()
            if self.provider == "mock":
                content, usage = self._call_mock(task=task, context=context)
            elif self.provider == "openai_compatible":
                augmented_user_prompt = user_prompt
                if parse_attempt > 0:
                    augmented_user_prompt = (
                        user_prompt
                        + "\n\nIMPORTANT: Return strictly valid JSON object only."
                    )
                content, usage = self._call_openai_compatible(
                    system_prompt, augmented_user_prompt
                )
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")

            latency = int((time.time() - start) * 1000)
            usage.latency_ms = usage.latency_ms or []
            usage.latency_ms.append(latency)
            last_content = content
            _accumulate_usage(total_usage, usage)

            try:
                parsed = parse_llm_json(content)
                return LLMResult(content=content, parsed_json=parsed, usage=total_usage)
            except JsonParseError as exc:
                last_error = exc
                if parse_attempt == parse_attempts - 1:
                    raise
                continue

        # Defensive fallback; the loop should either return or raise.
        assert last_error is not None
        raise RuntimeError(
            f"LLM JSON parsing failed after retries: {last_error}. Last content: {last_content[:500]}"
        ) from last_error

    def _call_openai_compatible(self, system_prompt: str, user_prompt: str) -> tuple[str, LLMUsage]:
        api_key = self.api_key or os.getenv(self.api_key_env, "")
        if not api_key:
            raise ValueError(
                "Missing API key. Set `llm.api_key` or provide env var "
                f"'{self.api_key_env}'. Use provider=mock for local testing."
            )

        url = self.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
        }

        body: dict[str, Any] | None = None
        attempts = self.max_retries + 1
        current_url = url
        payload_bytes = json.dumps(payload).encode("utf-8")
        for attempt in range(attempts):
            req = urllib.request.Request(
                current_url,
                data=payload_bytes,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="ignore")
                if exc.code in {301, 302, 307, 308} and attempt < attempts - 1:
                    location = ""
                    try:
                        location = str(exc.headers.get("Location", "") or "").strip()
                    except Exception:
                        location = ""
                    if location:
                        current_url = location
                        continue
                if (
                    exc.code in {429, 500, 502, 503, 504}
                    and attempt < attempts - 1
                ):
                    time.sleep(self.retry_backoff_s * (2**attempt))
                    continue
                raise RuntimeError(f"LLM HTTP error {exc.code}: {detail}") from exc
            except urllib.error.URLError as exc:
                if attempt < attempts - 1:
                    time.sleep(self.retry_backoff_s * (2**attempt))
                    continue
                raise RuntimeError(f"LLM network error: {exc}") from exc
            except (IncompleteRead, ConnectionResetError, TimeoutError, OSError) as exc:
                if attempt < attempts - 1:
                    time.sleep(self.retry_backoff_s * (2**attempt))
                    continue
                raise RuntimeError(f"LLM transport error: {exc}") from exc

        if body is None:
            raise RuntimeError("LLM request failed without response body after retries")

        choices = body.get("choices", [])
        if not choices:
            raise RuntimeError(f"Invalid LLM response, missing choices: {body}")

        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))

        usage_info = body.get("usage", {})
        usage = LLMUsage(
            calls=1,
            prompt_tokens=usage_info.get("prompt_tokens"),
            completion_tokens=usage_info.get("completion_tokens"),
            total_tokens=usage_info.get("total_tokens"),
            latency_ms=[],
        )
        return content, usage

    def _call_mock(
        self,
        task: str = "",
        context: dict[str, Any] | None = None,
    ) -> tuple[str, LLMUsage]:
        payload = {
            "task": task,
            "mock": True,
            "context": context or {},
            "result": {},
        }
        if context and "mock_result" in context:
            payload["result"] = context["mock_result"]
        usage = LLMUsage(calls=1, latency_ms=[])
        return json.dumps(payload, ensure_ascii=False), usage


def _accumulate_usage(total: LLMUsage, delta: LLMUsage) -> None:
    total.calls += int(delta.calls or 0)
    total.prompt_tokens = _sum_opt(total.prompt_tokens, delta.prompt_tokens)
    total.completion_tokens = _sum_opt(total.completion_tokens, delta.completion_tokens)
    total.total_tokens = _sum_opt(total.total_tokens, delta.total_tokens)
    total.latency_ms = (total.latency_ms or []) + (delta.latency_ms or [])


def _sum_opt(a: int | None, b: int | None) -> int | None:
    if a is None and b is None:
        return None
    return int(a or 0) + int(b or 0)
