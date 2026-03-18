import json
import re
from dataclasses import dataclass


class JsonParseError(ValueError):
    """Raised when no valid JSON object can be extracted and parsed."""


@dataclass
class JsonExtractResult:
    raw: str
    extracted: str


def _extract_outer_json_object(text: str) -> str:
    start = text.find("{")
    if start == -1:
        raise JsonParseError("No JSON object found: missing '{'.")

    depth = 0
    in_str = False
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]

    raise JsonParseError("No complete JSON object found: unbalanced braces.")


def _iter_json_object_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for start in (idx for idx, ch in enumerate(text) if ch == "{"):
        depth = 0
        in_str = False
        escaped = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    chunk = text[start : idx + 1]
                    if chunk not in seen:
                        seen.add(chunk)
                        candidates.append(chunk)
                    break
                if depth < 0:
                    break
    return candidates


def _minimal_json_fix(text: str) -> str:
    fixed = text.strip()
    fixed = re.sub(r"^```(?:json)?\s*", "", fixed, flags=re.IGNORECASE)
    fixed = re.sub(r"\s*```$", "", fixed)
    fixed = fixed.replace("\ufeff", "")
    fixed = fixed.replace("\u201c", '"').replace("\u201d", '"')
    fixed = fixed.replace("\u2018", "'").replace("\u2019", "'")
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
    return fixed


def extract_json(text: str) -> JsonExtractResult:
    candidate = text.strip()
    if candidate.startswith("{") and candidate.endswith("}"):
        return JsonExtractResult(raw=text, extracted=candidate)
    candidates = _iter_json_object_candidates(text)
    if candidates:
        return JsonExtractResult(raw=text, extracted=candidates[0])
    extracted = _extract_outer_json_object(text)
    return JsonExtractResult(raw=text, extracted=extracted)


def parse_llm_json(text: str) -> dict:
    extraction = extract_json(text)
    candidates: list[str] = [extraction.extracted]
    for chunk in _iter_json_object_candidates(text):
        if chunk != extraction.extracted:
            candidates.append(chunk)

    last_error: Exception | None = None
    for candidate in candidates:
        for payload in (candidate, _minimal_json_fix(candidate)):
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError as exc:
                last_error = exc
                continue
            if isinstance(obj, dict):
                return obj
            last_error = JsonParseError("Parsed JSON payload is not an object.")

    if last_error is None:
        raise JsonParseError("JSON parsing failed: no candidate object found.")
    raise JsonParseError(f"JSON parsing failed after minimal fix: {last_error}") from last_error
