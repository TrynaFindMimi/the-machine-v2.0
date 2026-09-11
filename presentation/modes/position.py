from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from config.palette import BLACK, WHITE
from config.strings import POSITION_TITLE
from core.handedness import get_handedness
from core.results import get_gesture, to_pixel_points
from presentation.ui.drawing import draw_bbox, draw_landmarks, draw_skeleton, spaced
from presentation.ui.effects import draw_viewfinder_crosshair
from presentation.ui.theme import FONT


def _put_text_box(
    frame: NDArray[np.uint8],
    text: str,
    org: tuple[int, int],
    font_scale: float,
    thickness: int,
    color: tuple[int, int, int] = WHITE,
    bg: tuple[int, int, int] = BLACK,
    pad_x: int = 6,
    pad_y: int = 4,
) -> None:
    (tw, th), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)
    x, y = org
    x1, y1 = x - pad_x, y - th - pad_y
    x2, y2 = x + tw + pad_x, y + baseline + pad_y // 2
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), bg, -1)
    cv2.putText(frame, text, (x, y), FONT, font_scale, color, thickness, cv2.LINE_AA)


def draw(frame: NDArray[np.uint8], results) -> tuple[NDArray[np.uint8], int]:
    _put_text_box(frame, spaced(POSITION_TITLE), (10, 22), 0.70, 2, WHITE, BLACK)
    h, w = frame.shape[:2]
    hands = results.hand_landmarks if results.hand_landmarks else []
    hand_count = len(hands)
    if hand_count == 0:
        draw_viewfinder_crosshair(frame, WHITE)
    for i, hand_landmarks in enumerate(hands[:2]):
        pts = to_pixel_points(hand_landmarks, w, h)
        gesture_name, gesture_score = get_gesture(results, i)
        draw_bbox(frame, pts, WHITE, thickness=2)
        draw_skeleton(frame, pts)
        draw_landmarks(frame, pts)
        hand_label = get_handedness(results, i)
        x1 = min(p[0] for p in pts) - 10
        y1 = min(p[1] for p in pts) - 10
        _put_text_box(
            frame,
            hand_label,
            (max(0, x1), max(18, y1 - 10)),
            0.60,
            2,
            WHITE,
            BLACK,
            pad_x=5,
            pad_y=3,
        )
        _put_text_box(
            frame,
            f"{gesture_name} {gesture_score:.0%}",
            (max(0, x1), max(18, y1 - 38)),
            0.65,
            2,
            WHITE,
            BLACK,
            pad_x=5,
            pad_y=3,
        )
    y_off = 78
    for i in range(min(hand_count, 2)):
        gesture_name, _ = get_gesture(results, i)
        label = get_handedness(results, i)
        _put_text_box(
            frame,
            f"{label}: {gesture_name}",
            (10, y_off),
            0.60,
            2,
            WHITE,
            BLACK,
            pad_x=6,
            pad_y=4,
        )
        y_off += 30
    return frame, hand_count
