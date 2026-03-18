from maner.core.dataset import GenericJSONLReader
from maner.core.schema import load_schema


def test_schema_load() -> None:
    schema = load_schema("tests/fixtures/schema_example.json")
    assert "PERSON" in schema.entity_type_names
    assert len(schema.relation_constraints) == 2


def test_generic_reader() -> None:
    reader = GenericJSONLReader("tests/fixtures/tiny_data.jsonl")
    samples = list(reader.iter_samples())
    assert len(samples) == 2
    assert samples[0].sample_id == "s1"
    assert samples[0].gold_mentions is not None
