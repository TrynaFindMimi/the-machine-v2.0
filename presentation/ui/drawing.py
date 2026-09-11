from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from config.palette import WHITE

HAND_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (1, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]


def spaced(text: str) -> str:
    return " ".join(text.upper())


def draw_skeleton(frame: NDArray[np.uint8], pts: Sequence[tuple[int, int]]) -> None:
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], WHITE, 2, cv2.LINE_AA)


def draw_landmarks(frame: NDArray[np.uint8], pts: Sequence[tuple[int, int]]) -> None:
    for i, pt in enumerate(pts):
        cv2.circle(frame, pt, 4, WHITE, -1, cv2.LINE_AA)
        label = str(i)
        x, y = pt[0] + 7, pt[1] - 7
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x - 1, y - th - 1), (x + tw + 1, y + 1), (0, 0, 0), -1)
        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)


def draw_bbox(
    frame: NDArray[np.uint8],
    pts: Sequence[tuple[int, int]],
    color: tuple[int, int, int],
    thickness: int = 1,
    corner_len: int = 12,
) -> tuple[int, int, int, int]:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    pad = 10
    x1, y1 = max(0, min(xs) - pad), max(0, min(ys) - pad)
    x2, y2 = max(xs) + pad, max(ys) + pad
    h, w = frame.shape[:2]
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    cl = corner_len
    t = thickness + 1
    for cx, cy, dx, dy in [
        (x1, y1, 1, 1),
        (x2, y1, -1, 1),
        (x1, y2, 1, -1),
        (x2, y2, -1, -1),
    ]:
        cv2.line(frame, (cx, cy), (cx + dx * cl, cy), color, t, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * cl), color, t, cv2.LINE_AA)
    return x1, y1, x2, y2


def draw_line(
    frame: NDArray[np.uint8],
    p1: Sequence[int] | NDArray[np.generic],
    p2: Sequence[int] | NDArray[np.generic],
    color: tuple[int, int, int],
    thickness: int = 1,
) -> tuple[tuple[int, int], tuple[int, int]]:
    pt1 = (int(p1[0]), int(p1[1]))
    pt2 = (int(p2[0]), int(p2[1]))
    cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)
    return pt1, pt2
