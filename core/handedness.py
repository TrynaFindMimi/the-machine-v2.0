from __future__ import annotations

from typing import Final, Protocol, Sequence


class _Category(Protocol):
    category_name: str


class _Results(Protocol):
    handedness: Sequence[Sequence[_Category]] | None


LEFT: Final = "Left"
RIGHT: Final = "Right"
UNKNOWN: Final = "?"


def normalize_handedness(raw_label: str) -> str:
    if raw_label == "Left":
        return RIGHT
    if raw_label == "Right":
        return LEFT
    return UNKNOWN


def get_handedness(results: _Results, idx: int) -> str:
    if not results.handedness or idx >= len(results.handedness):
        return UNKNOWN
    cats = results.handedness[idx]
    if not cats:
        return UNKNOWN
    raw = cats[0].category_name
    return normalize_handedness(str(raw))


def is_left(results: _Results, idx: int) -> bool:
    return get_handedness(results, idx) == LEFT


def is_right(results: _Results, idx: int) -> bool:
    return get_handedness(results, idx) == RIGHT
