import pytest

from maner.cli.run_pipeline import _apply_overrides


def test_apply_overrides_sets_nested_values() -> None:
    cfg = {"data": {"data_path": "a.jsonl"}, "pipeline": {"max_debate_turns": 5}}
    _apply_overrides(
        cfg,
        [
            "data.data_path=outputs/bc2gm/bc2gm_test.20.jsonl",
            "pipeline.max_debate_turns=3",
            "llm.temperature=0.2",
            "flags.enabled=true",
        ],
    )
    assert cfg["data"]["data_path"] == "outputs/bc2gm/bc2gm_test.20.jsonl"
    assert cfg["pipeline"]["max_debate_turns"] == 3
    assert cfg["llm"]["temperature"] == 0.2
    assert cfg["flags"]["enabled"] is True


def test_apply_overrides_requires_equal_sign() -> None:
    cfg = {}
    with pytest.raises(ValueError):
        _apply_overrides(cfg, ["pipeline.max_debate_turns"])
