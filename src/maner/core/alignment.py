from __future__ import annotations

from typing import Iterable

from maner.core.types import is_valid_offsets


def align_substring_offsets(
    text: str,
    quote: str,
    start_hint: int | None = None,
    end_hint: int | None = None,
) -> tuple[int, int] | None:
    """Align a quote to offsets on the same raw string, preferring nearest start_hint."""
    if not quote:
        return None

    if (
        start_hint is not None
        and end_hint is not None
        and is_valid_offsets(text, start_hint, end_hint)
        and text[start_hint:end_hint] == quote
    ):
        return start_hint, end_hint

    starts = list(_find_all_starts(text, quote))
    if not starts:
        return None

    if start_hint is None:
        start = starts[0]
    else:
        start = min(starts, key=lambda s: abs(s - start_hint))
    return start, start + len(quote)


def _find_all_starts(text: str, quote: str) -> Iterable[int]:
    pos = text.find(quote)
    while pos >= 0:
        yield pos
        pos = text.find(quote, pos + 1)

