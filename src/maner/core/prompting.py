from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import yaml


class PromptManager:
    def __init__(self, prompt_path: str | Path | Iterable[str | Path]):
        self.templates: dict[str, Any] = {}
        for path in _normalize_prompt_paths(prompt_path):
            with path.open("r", encoding="utf-8") as f:
                payload = yaml.safe_load(f) or {}
            self.templates = _deep_merge_prompts(self.templates, payload)

    def render(self, name: str, **kwargs: Any) -> tuple[str, str]:
        if name not in self.templates:
            raise KeyError(f"Prompt template not found: {name}")
        template = self.templates[name]
        system_t = str(template.get("system", ""))
        user_t = str(template.get("user", ""))

        serialized = {
            k: (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))
            for k, v in kwargs.items()
        }
        system = system_t.format_map(serialized)
        user = user_t.format_map(serialized)
        return system, user


def _normalize_prompt_paths(
    prompt_path: str | Path | Iterable[str | Path],
) -> list[Path]:
    if isinstance(prompt_path, (str, Path)):
        return [Path(prompt_path)]
    return [Path(p) for p in prompt_path]


def _deep_merge_prompts(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_prompts(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged
