from pathlib import Path

from maner.core.config import load_yaml
from maner.core.types import ExpertConstraints, UsageCost
from maner.orchestrator.pipeline import PipelineRunner, write_predictions


def test_pipeline_smoke_end_to_end(tmp_path) -> None:
    cfg = load_yaml("configs/default.yaml")
    cfg["output"] = {"predictions_path": str(tmp_path / "pred.jsonl")}
    cfg["memory"]["sqlite_path"] = str(tmp_path / "memory.db")

    runner = PipelineRunner(cfg)
    try:
        records = runner.run()
    finally:
        runner.close()

    assert records
    assert "mentions" in records[0]
    assert "traces" in records[0]
    assert "costs" in records[0]

    out_path = Path(cfg["output"]["predictions_path"])
    write_predictions(records, out_path)
    assert out_path.exists()


def test_pipeline_span_augmentation_from_expert_trace(tmp_path, monkeypatch) -> None:
    cfg = load_yaml("configs/default.yaml")
    cfg["output"] = {"predictions_path": str(tmp_path / "pred_aug.jsonl")}
    cfg["memory"]["sqlite_path"] = str(tmp_path / "memory_aug.db")
    cfg["pipeline"]["allow_expert_span_augmentation"] = True
    cfg["pipeline"]["allow_re_span_augmentation"] = False
    cfg["pipeline"]["rerun_after_span_augmentation"] = False

    runner = PipelineRunner(cfg)
    try:
        def _fake_expert_run(**kwargs):
            text = kwargs["text"]
            return (
                ExpertConstraints(),
                UsageCost(),
                {
                    "agent": "expert",
                    "raw": "{}",
                    "parsed": {},
                    "span_proposals": [
                        {
                            "source": "expert",
                            "start": 11,
                            "end": 13,
                            "text": text[11:13],
                            "confidence": 0.8,
                            "evidence": [{"quote": text[11:13], "start": 11, "end": 13}],
                            "rationale": "test_new_span",
                            "type_hints": [],
                        }
                    ],
                },
            )

        monkeypatch.setattr(runner.expert_agent, "run", _fake_expert_run)
        records = runner.run()
    finally:
        runner.close()

    assert records
    aug_trace = records[0]["traces"]["span_augmentation"]
    assert aug_trace["enabled"] is True
    assert aug_trace["added_count"] >= 1
