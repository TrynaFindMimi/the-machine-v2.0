from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from config.palette import BLUE, WHITE
from config.settings import PERCEPTRON
from config.strings import LINE_TITLE
from core.perceptron import Perceptron, build_dataset
from core.results import count_hands, to_pixel_points
from presentation.ui.drawing import draw_bbox, draw_landmarks, draw_line, draw_skeleton, spaced
from presentation.ui.effects import draw_viewfinder_crosshair
from presentation.ui.theme import FONT

P_A: int = PERCEPTRON.point_a_idx
P_B: int = PERCEPTRON.point_b_idx
EPOCH_BUDGET: int = PERCEPTRON.epoch_budget
_FINGERS_THRESH: float = PERCEPTRON.fingers_together_thresh

_perceptrons: list[Perceptron | None] = [None, None]


def reset_perceptrons() -> None:
    _perceptrons[0] = None
    _perceptrons[1] = None


def draw(frame: NDArray[np.uint8], results) -> tuple[NDArray[np.uint8], int]:
    cv2.putText(frame, spaced(LINE_TITLE), (10, 18), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
    h, w = frame.shape[:2]
    hands = results.hand_landmarks
    hand_count = count_hands(results)
    if hand_count == 0:
        draw_viewfinder_crosshair(frame, WHITE)
        reset_perceptrons()
        return frame, hand_count
    if not hands:
        return frame, hand_count
    for i, lm in enumerate(hands[:2]):
        pts = to_pixel_points(lm, w, h)
        draw_bbox(frame, pts, WHITE)
        draw_skeleton(frame, pts)
        draw_landmarks(frame, pts)
        p5 = np.array([lm[P_A].x, lm[P_A].y], dtype=float)
        p9 = np.array([lm[P_B].x, lm[P_B].y], dtype=float)
        if float(np.linalg.norm(p9 - p5)) < _FINGERS_THRESH:
            _perceptrons[i] = None
            continue
        data = build_dataset(p5, p9)
        if data[0] is None:
            continue
        X, y = data
        if _perceptrons[i] is None:
            _perceptrons[i] = Perceptron()
        _perceptrons[i].train_budget(X, y, EPOCH_BUDGET)  # type: ignore[union-attr]
        draw_line(frame, pts[P_A], pts[P_B], color=BLUE, thickness=1)
    return frame, hand_count
