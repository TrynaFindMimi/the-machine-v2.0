from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from config.palette import BASALT, BRONZE, GOLD, IVORY


def draw_panel(
    frame: NDArray[np.uint8],
    x: int,
    y: int,
    w: int,
    h: int,
    fill: tuple[int, int, int] = BASALT,
    border: tuple[int, int, int] = BRONZE,
    thickness: int = 1,
) -> None:
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(frame.shape[1] - 1, x + w), min(frame.shape[0] - 1, y + h)
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(frame, (x1, y1), (x2, y2), fill, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), border, thickness, cv2.LINE_AA)


def draw_fret_band(
    frame: NDArray[np.uint8],
    y: int,
    x0: int,
    x1: int,
    color: tuple[int, int, int] = BRONZE,
    unit: int = 16,
    thickness: int = 2,
) -> None:
    if x1 <= x0:
        return
    cv2.line(frame, (x0, y), (x1, y), color, thickness, cv2.LINE_AA)
    x = x0
    while x < x1 - unit:
        half = (unit * 2) // 5
        cv2.line(frame, (x, y), (x, y - unit), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x, y - unit), (x + half, y - unit), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x + half, y - unit), (x + half, y), color, thickness, cv2.LINE_AA)
        x += unit
    cv2.line(frame, (x0, y - unit), (x1, y - unit), color, thickness, cv2.LINE_AA)


def draw_pediment(
    frame: NDArray[np.uint8],
    cx: int,
    top: int,
    width: int,
    color: tuple[int, int, int] = GOLD,
    thickness: int = 2,
) -> None:
    half = width // 2
    pts = np.array(
        [[cx - half, top + half], [cx, top], [cx + half, top + half]],
        dtype=np.int32,
    )
    cv2.polylines(frame, [pts], True, color, thickness, cv2.LINE_AA)
    cv2.line(frame, (cx - half, top + half), (cx + half, top + half), color, thickness, cv2.LINE_AA)


def draw_laurel_divider(
    frame: NDArray[np.uint8],
    y: int,
    x0: int,
    x1: int,
    color: tuple[int, int, int] = BRONZE,
    thickness: int = 1,
) -> None:
    cv2.line(frame, (x0, y), (x1, y), color, thickness, cv2.LINE_AA)
    for x in range(x0 + 16, x1, 32):
        cv2.line(frame, (x - 4, y - 4), (x, y), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x, y), (x + 4, y - 4), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x - 4, y + 4), (x, y), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x, y), (x + 4, y + 4), color, thickness, cv2.LINE_AA)


def draw_text_centered(
    frame: NDArray[np.uint8],
    text: str,
    cx: int,
    y: int,
    font_scale: float,
    thickness: int,
    color: tuple[int, int, int] = IVORY,
    font=cv2.FONT_HERSHEY_SIMPLEX,
) -> None:
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.putText(
        frame,
        text,
        (max(0, cx - tw // 2), y),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
