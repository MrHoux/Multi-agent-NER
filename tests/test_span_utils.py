from maner.core.types import Span, merge_spans, span_iou, span_overlap


def test_span_overlap_and_iou() -> None:
    a = Span(text="A", start=0, end=4)
    b = Span(text="B", start=2, end=6)
    assert span_overlap(a, b) == 2
    assert abs(span_iou(a, b) - (2 / 6)) < 1e-6


def test_merge_spans() -> None:
    spans = [
        Span(text="x", start=0, end=3),
        Span(text="y", start=2, end=5),
        Span(text="z", start=8, end=10),
    ]
    merged = merge_spans(spans)
    assert len(merged) == 2
    assert merged[0].start == 0
    assert merged[0].end == 5
