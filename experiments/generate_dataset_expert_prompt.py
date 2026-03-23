from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml

from maner.core.config import load_yaml
from maner.core.schema import load_schema
from maner.llm.client import LLMClient


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_profile(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = repo_root / path
    return path


def _load_template(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    template = payload.get("prompt_designer", {})
    return {
        "system": str(template.get("system", "")),
        "user": str(template.get("user", "")),
    }


def _load_base_expert_prompts(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    prompts: dict[str, dict[str, str]] = {}
    for task_name in ("expert_retrieval_agent", "expert_agent"):
        item = payload.get(task_name, {})
        if not isinstance(item, dict):
            raise ValueError(f"Missing {task_name} in {path}")
        system = str(item.get("system", "")).strip()
        user = str(item.get("user", "")).strip()
        if not system or not user:
            raise ValueError(f"Incomplete {task_name} prompt in {path}")
        prompts[task_name] = {"system": system, "user": user}
    return prompts


def _design_goals() -> list[str]:
    return [
        "Drive type decisions from schema semantics, local context, and explicit textual evidence.",
        "Treat schema type descriptions as the authoritative semantic contract in zero-shot mode.",
        "Explain ambiguity resolution at a conceptual level instead of naming exact strings or memorized examples.",
        "Encourage conservative abstention when evidence is weak or conflicting.",
        "Allow boundary updates only when the text itself supports the adjustment.",
        "Preserve portability to unseen samples; avoid dataset-specific lexical lookup behavior.",
        "Forbid exact keyword matching, field matching, blacklist rules, whitelist rules, and copied examples.",
    ]

def _find_style_violations(guidance: list[str]) -> list[str]:
    violations: list[str] = []
    markers = ("e.g.", "for example", "for instance", "such as")
    for item in guidance:
        lowered = item.lower()
        if any(marker in lowered for marker in markers):
            violations.append("illustrative example marker")
        if any(ch in item for ch in ("'", '"', "`")):
            violations.append("quoted or literal token")
    deduped: list[str] = []
    seen: set[str] = set()
    for item in violations:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _sanitize_guidance_item(item: str) -> str:
    text = item.replace("`", "").replace('"', "").replace("'", "").strip()
    lowered = text.lower()
    for marker in ("e.g.", "for example", "for instance", "such as"):
        idx = lowered.find(marker)
        if idx >= 0:
            text = text[:idx].rstrip(" ,;:-")
            lowered = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text


def _coerce_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        parts = [
            re.sub(r"^[\-\*\d\.\)\s]+", "", line).strip()
            for line in value.splitlines()
        ]
        normalized = [item for item in parts if item]
        if normalized:
            return normalized
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _parse_generated_prompt(payload: dict[str, Any]) -> tuple[list[str], list[str]]:
    guidance: list[str] = []
    for key in (
        "expert_guidance",
        "guidance",
        "expert_bullets",
        "prompt_points",
        "bullets",
    ):
        guidance = _coerce_string_list(payload.get(key))
        if guidance:
            break
    if not guidance:
        for value in payload.values():
            guidance = _coerce_string_list(value)
            if len(guidance) >= 4:
                break
    if not guidance:
        raise ValueError("Generator output missing `expert_guidance`")
    raw_summary = payload.get("design_summary", payload.get("summary", []))
    summary = _coerce_string_list(raw_summary)
    return guidance, summary


def _stabilize_guidance(guidance: list[str], schema_type_names: list[str]) -> list[str]:
    lowered_types = {item.lower() for item in schema_type_names}
    stabilized: list[str] = [
        "Treat entity mentions as identity-bearing references grounded in the text; do not label pronouns, bare role nouns, common descriptors, dates, or pure numerals unless the schema explicitly targets them.",
        "Prefer low confidence or abstention when evidence does not support a stable named-entity interpretation.",
    ]
    if {"per", "org", "loc"} & lowered_types:
        stabilized.append(
            "Resolve person, organization, and location ambiguity from referential function in context: individual human actor, organized collective actor, or geographic place."
        )
    if any(
        token in lowered_types
        for token in {"misc", "miscellaneous", "other", "others", "residual"}
    ) or any(
        any(token in item for token in ("misc", "other", "residual"))
        for item in lowered_types
    ):
        stabilized.append(
            "Use residual or catch-all labels only as a last resort after excluding more specific schema types, and require strong text-grounded evidence that the mention denotes a named category rather than a generic modifier."
        )
    stabilized.extend(guidance)

    deduped: list[str] = []
    seen: set[str] = set()
    for item in stabilized:
        cleaned = _sanitize_guidance_item(item)
        key = re.sub(r"\s+", " ", cleaned.strip().lower())
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(cleaned)
    return deduped


def _build_schema_semantics(schema: Any) -> dict[str, Any]:
    entity_types: list[dict[str, str]] = []
    lowered_types = {item.lower() for item in schema.entity_type_names}
    derived_notes: list[str] = [
        "Zero-shot mode: use only dataset name, type names, and type descriptions as semantic guidance.",
        "All guidance must stay abstract and portable; no examples or lexical lookup rules are allowed.",
    ]
    for item in schema.entity_types:
        entity_types.append(
            {
                "name": item.name,
                "description": item.description,
            }
        )
    if any(token in lowered_types for token in {"misc", "miscellaneous", "other", "others", "residual"}):
        derived_notes.append(
            "Residual labels should be described as last-resort categories with stricter evidence requirements."
        )
    if {"per", "org", "loc"} & lowered_types:
        derived_notes.append(
            "Where relevant, distinguish people, organizations, and locations by referential function rather than surface form."
        )
    return {
        "entity_types": entity_types,
        "derived_notes": derived_notes,
    }


def _build_overlay_from_guidance(
    base_prompts: dict[str, dict[str, str]],
    guidance: list[str],
) -> dict[str, dict[str, str]]:
    guidance_block = "Dataset-specific expert guidance (mandatory):\n" + "\n".join(
        f"- {item}" for item in guidance
    )
    anchor = "Collaboration protocol with RAG (mandatory):"
    overlay: dict[str, dict[str, str]] = {}
    for task_name, base_prompt in base_prompts.items():
        user = base_prompt["user"]
        if anchor in user:
            user = user.replace(anchor, guidance_block + "\n\n" + anchor, 1)
        else:
            user = user + "\n\n" + guidance_block
        overlay[task_name] = {
            "system": base_prompt["system"],
            "user": user,
        }
    return overlay


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dataset-specific expert_agent prompt")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    repo_root = _repo_root()
    profile_path = _resolve(repo_root, args.profile)
    profile = _load_profile(profile_path)
    generation_cfg = dict(profile.get("prompt_generation", {}))
    if not generation_cfg:
        raise ValueError("Profile missing `prompt_generation` section")

    output_path = _resolve(repo_root, str(generation_cfg["output_prompt_path"]))
    metadata_path = _resolve(repo_root, str(generation_cfg["metadata_path"]))
    if output_path.exists() and not args.force:
        print(json.dumps({"generated": False, "reason": "exists", "output_path": str(output_path)}, ensure_ascii=False))
        return

    llm_cfg_path = _resolve(repo_root, str(generation_cfg["llm_config_path"]))
    llm_cfg = load_yaml(llm_cfg_path).get("llm", {})
    llm = LLMClient(llm_cfg)

    schema_path = _resolve(repo_root, str(generation_cfg["schema_path"]))
    schema = load_schema(schema_path)
    base_prompt_path = _resolve(repo_root, str(generation_cfg.get("base_prompt_path", "configs/prompts_cot.yaml")))
    base_expert_prompts = _load_base_expert_prompts(base_prompt_path)

    template_path = _resolve(
        repo_root,
        str(generation_cfg.get("template_path", "configs/expert_prompt_generator.yaml")),
    )
    template = _load_template(template_path)

    dataset_label = str(schema.dataset_name or profile.get("dataset_id", "")).strip()

    serialized_schema = json.dumps(schema.to_prompt_block(), ensure_ascii=False, indent=2)
    serialized_schema_semantics = json.dumps(_build_schema_semantics(schema), ensure_ascii=False, indent=2)
    serialized_goals = json.dumps(_design_goals(), ensure_ascii=False, indent=2)
    serialized_base_prompt = json.dumps(base_expert_prompts, ensure_ascii=False, indent=2)

    extra_instruction = "None."
    last_issues: list[str] = []
    generated_prompt: dict[str, dict[str, str]] | None = None
    design_summary: list[str] = []
    last_guidance: list[str] = []

    for _attempt in range(4):
        system_prompt = template["system"]
        user_prompt = template["user"].format_map(
            {
                "dataset_id": profile.get("dataset_id", ""),
                "dataset_name": dataset_label,
                "entity_schema": serialized_schema,
                "schema_semantics": serialized_schema_semantics,
                "design_goals": serialized_goals,
                "base_expert_prompt": serialized_base_prompt,
                "extra_instruction": extra_instruction,
            }
        )
        result = llm.chat_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            task="generate_expert_prompt",
            context=None,
        )
        try:
            guidance, summary = _parse_generated_prompt(result.parsed_json)
        except ValueError as exc:
            last_issues = [str(exc)]
            extra_instruction = (
                "Return a JSON object with keys `design_summary` and `expert_guidance`. "
                "`expert_guidance` must be an array of abstract strings only."
            )
            continue
        guidance = _stabilize_guidance(guidance, schema.entity_type_names)
        last_guidance = guidance
        style_violations = _find_style_violations(guidance)
        prompt_block = _build_overlay_from_guidance(base_expert_prompts, guidance)
        if not style_violations:
            generated_prompt = prompt_block
            design_summary = summary
            break
        issue_notes: list[str] = []
        if style_violations:
            issue_notes.append(
                "Do not include illustrative example markers or quoted literal tokens."
            )
        last_issues = style_violations
        extra_instruction = " ".join(issue_notes) + " Keep all guidance abstract."

    if generated_prompt is None:
        raise RuntimeError(
            "Failed to generate leak-free expert prompt. "
            + (f"Last detected issues: {', '.join(last_issues)}. " if last_issues else "")
            + (f"Last guidance: {json.dumps(last_guidance, ensure_ascii=False)}" if last_guidance else "")
        )

    overlay = generated_prompt
    metadata = {
        "dataset_id": profile.get("dataset_id", ""),
        "dataset_name": dataset_label,
        "schema_path": str(schema_path),
        "generation_mode": "zero_shot_schema_only",
        "design_summary": design_summary,
    }
    _write_yaml(output_path, overlay)
    _write_json(metadata_path, metadata)
    print(
        json.dumps(
            {
                "generated": True,
                "output_path": str(output_path),
                "metadata_path": str(metadata_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
