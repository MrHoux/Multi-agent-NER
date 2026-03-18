from maner.core.schema import load_schema
from maner.eval.metrics import evaluate_from_files


def test_strict_metrics_micro_macro() -> None:
    schema = load_schema("tests/fixtures/schema_example.json")
    res = evaluate_from_files(
        gold_path="tests/fixtures/tiny_gold.jsonl",
        pred_path="tests/fixtures/tiny_pred_eval.jsonl",
        schema=schema,
    )
    assert 0.0 <= res["micro"]["f1"] <= 1.0
    assert 0.0 <= res["macro"]["f1"] <= 1.0
    assert "PERSON" in res["per_type"]
    assert res["errors"]["type_mismatch"] >= 1
