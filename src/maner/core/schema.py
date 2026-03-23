from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EntityTypeDef:
    name: str
    description: str


@dataclass
class SchemaDefinition:
    entity_types: list[EntityTypeDef]
    dataset_name: str = ""
    relation_constraints: list[dict] = field(default_factory=list)

    @property
    def entity_type_names(self) -> list[str]:
        return [t.name for t in self.entity_types]

    def to_prompt_block(self) -> list[dict[str, str]]:
        return [{"name": t.name, "description": t.description} for t in self.entity_types]


def load_schema(schema_path: str | Path) -> SchemaDefinition:
    path = Path(schema_path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if "entity_types" not in raw or not isinstance(raw["entity_types"], list):
        raise ValueError("schema.json must include list field 'entity_types'.")

    entity_types = []
    for item in raw["entity_types"]:
        name = str(item.get("name", "")).strip()
        description = str(item.get("description", "")).strip()
        if not name:
            raise ValueError("schema.json entity type item missing 'name'.")
        entity_types.append(EntityTypeDef(name=name, description=description))

    relation_constraints = raw.get("relation_constraints", [])
    if relation_constraints is None:
        relation_constraints = []
    if not isinstance(relation_constraints, list):
        raise ValueError("schema.json field 'relation_constraints' must be a list if provided.")

    dataset_name = str(raw.get("dataset_name") or raw.get("dataset_id") or "").strip()

    return SchemaDefinition(
        dataset_name=dataset_name,
        entity_types=entity_types,
        relation_constraints=relation_constraints,
    )
