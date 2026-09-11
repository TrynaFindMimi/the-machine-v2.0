from __future__ import annotations

from typing import Protocol, Sequence


class _Category(Protocol):
    category_name: str
    score: float


class _Landmark(Protocol):
    x: float
    y: float


class _Results(Protocol):
    gestures: Sequence[Sequence[_Category]] | None
    hand_landmarks: Sequence[Sequence[_Landmark]] | None
    handedness: Sequence[Sequence[_Category]] | None


def get_gesture(results: _Results, idx: int) -> tuple[str, float]:
    from config.strings import GESTURE_NONE

    if not results.gestures or idx >= len(results.gestures):
        return GESTURE_NONE, 0.0
    cats = results.gestures[idx]
    if not cats:
        return GESTURE_NONE, 0.0
    top = cats[0]
    return str(top.category_name), float(top.score)


def to_pixel_points(hand_landmarks: Sequence[_Landmark], w: int, h: int) -> list[tuple[int, int]]:
    return [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]


def count_hands(results: _Results) -> int:
    return len(results.hand_landmarks) if results.hand_landmarks else 0
