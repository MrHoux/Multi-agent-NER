from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


class PromptManager:
    def __init__(self, prompt_path: str | Path):
        with Path(prompt_path).open("r", encoding="utf-8") as f:
            self.templates = yaml.safe_load(f) or {}

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
