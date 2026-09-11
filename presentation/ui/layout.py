from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from config.palette import BASALT, BRONZE, GOLD, IVORY, OLIVE, STONE
from config.strings import MODES_ORDER
from presentation.ui.greek import draw_fret_band
from presentation.ui.theme import FONT, SIDEBAR_W


def draw_sidebar(frame: NDArray[np.uint8], current_mode: str, hand_count: int, fps: float) -> None:
    h, w = frame.shape[:2]
    sx = w - SIDEBAR_W
    overlay = frame.copy()
    cv2.rectangle(overlay, (sx, 0), (w, h), BASALT, -1)
    cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
    cv2.line(frame, (sx, 0), (sx, h), BRONZE, 1, cv2.LINE_AA)
    draw_fret_band(frame, 22, sx + 8, w - 8, BRONZE, unit=14, thickness=1)
    y = 58
    cv2.putText(frame, "MODE", (sx + 14, y), FONT, 0.4, GOLD, 1, cv2.LINE_AA)
    y += 20
    for mode in MODES_ORDER:
        is_active = mode == current_mode
        text_color = IVORY if is_active else STONE
        dot_color = GOLD if is_active else BRONZE
        cv2.circle(frame, (sx + 18, y - 3), 3, dot_color, -1, cv2.LINE_AA)
        cv2.putText(frame, mode.upper(), (sx + 28, y), FONT, 0.4, text_color, 1, cv2.LINE_AA)
        y += 20
    y += 10
    cv2.line(frame, (sx + 10, y), (w - 10, y), BRONZE, 1, cv2.LINE_AA)
    y += 18
    cv2.putText(frame, "HANDS", (sx + 14, y), FONT, 0.35, GOLD, 1, cv2.LINE_AA)
    y += 20
    count_color = OLIVE if hand_count > 0 else STONE
    cv2.putText(frame, str(hand_count), (sx + 14, y), FONT, 1.0, count_color, 1, cv2.LINE_AA)
    y += 26
    cv2.line(frame, (sx + 10, y), (w - 10, y), BRONZE, 1, cv2.LINE_AA)
    y += 18
    cv2.putText(frame, "FPS", (sx + 14, y), FONT, 0.35, GOLD, 1, cv2.LINE_AA)
    y += 20
    cv2.putText(frame, f"{fps:05.1f}", (sx + 14, y), FONT, 0.5, IVORY, 1, cv2.LINE_AA)
    y += 24
    cv2.line(frame, (sx + 10, y), (w - 10, y), BRONZE, 1, cv2.LINE_AA)
    y += 18
    cv2.putText(frame, "CONTROLS", (sx + 14, y), FONT, 0.35, GOLD, 1, cv2.LINE_AA)
    y += 20
    cv2.putText(frame, "N  next", (sx + 14, y), FONT, 0.4, STONE, 1, cv2.LINE_AA)
    y += 16
    cv2.putText(frame, "Q  quit", (sx + 14, y), FONT, 0.4, STONE, 1, cv2.LINE_AA)
