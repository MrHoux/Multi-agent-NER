import re
from pathlib import Path
from typing import Any

import yaml


def _resolve_env_placeholders(value: Any) -> Any:
    if isinstance(value, str):
        pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
        return pattern.sub(lambda m: __import__("os").environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _resolve_env_placeholders(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_placeholders(v) for v in value]
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return _resolve_env_placeholders(raw)
