from pathlib import Path
from typing import Any

import yaml
from datetime import datetime
from maner.core.config import load_yaml
from maner.orchestrator.pipeline import PipelineRunner, write_predictions


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run MANER multi-agent NER pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config value, e.g. --set data.data_path=datasets/my_data.jsonl",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    _apply_overrides(cfg, args.overrides or [])

    output_cfg = cfg.get("output", {})
    pred_path = output_cfg.get("predictions_path")
    if not pred_path:
        output_dir = Path(cfg.get("output_dir", "outputs"))
        pred_path = str(output_dir / "predictions.jsonl")

    runner = PipelineRunner(cfg)
    try:
        records = runner.run()
    finally:
        runner.close()

    write_predictions(records, pred_path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [pipeline] predictions_written | path={pred_path}")


def _apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> None:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid --set override (missing '='): {item}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --set override (empty key): {item}")
        value = yaml.safe_load(raw_value)
        _set_nested(cfg, key, value)


def _set_nested(root: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = [p.strip() for p in dotted_key.split(".") if p.strip()]
    if not parts:
        raise ValueError(f"Invalid override key: {dotted_key}")

    cur: dict[str, Any] = root
    for part in parts[:-1]:
        node = cur.get(part)
        if not isinstance(node, dict):
            node = {}
            cur[part] = node
        cur = node
    cur[parts[-1]] = value


if __name__ == "__main__":
    main()
