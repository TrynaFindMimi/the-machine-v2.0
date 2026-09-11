from __future__ import annotations

import time

import cv2
import numpy as np
from numpy.typing import NDArray

from config.palette import BLACK, GRAY, GREEN, WHITE
from config.settings import MUSIC_ACTION_DISPLAY_SECONDS, MUSIC_EFFECTIVE_ZONE_FRACTION
from config.strings import (
    MUSIC_ACTION_FMT,
    MUSIC_COUNT_FMT,
    MUSIC_FINGERS_FMT,
    MUSIC_HAND_L,
    MUSIC_HAND_R,
    MUSIC_SAGA_FMT,
    MUSIC_SONG_FMT,
    MUSIC_STATE_FMT,
    MUSIC_TITLE,
    MUSIC_UNKNOWN,
    MUSIC_VOLUME_FMT,
    MUSIC_ZONE_TITLE,
)
from controllers.gestures import MusicGestureController
from controllers.volume import VolumeController
from core.fingers import (
    features_from_landmarks,
    hand_scale,
    index_thumb_distance,
)
from core.handedness import LEFT, RIGHT, get_handedness
from core.results import count_hands, to_pixel_points
from presentation.ui.drawing import draw_bbox, draw_skeleton, spaced
from presentation.ui.effects import draw_viewfinder_crosshair
from presentation.ui.theme import FONT

_gesture: MusicGestureController | None = None
_volume: VolumeController | None = None
_saga_name: str = MUSIC_UNKNOWN
_song_name: str = MUSIC_UNKNOWN
_track_text: str = MUSIC_UNKNOWN
_player_state: str = MUSIC_UNKNOWN
_player_volume: float = 0.0
_action_text: str = ""
_action_until: float = 0.0


def _get_gesture() -> MusicGestureController:
    global _gesture
    if _gesture is not None:
        return _gesture
    gc = MusicGestureController()
    gc.load_or_train()
    _gesture = gc
    return _gesture


def _get_volume() -> VolumeController:
    global _volume
    if _volume is not None:
        return _volume
    vc = VolumeController()
    vc.load_or_train()
    _volume = vc
    return _volume


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


def reset_state() -> None:
    _get_gesture().reset()
    _get_volume().reset()


def set_context(
    saga_name: str,
    song_name: str,
    track_text: str,
    player_state: str,
    volume: float = 0.0,
) -> None:
    global _saga_name, _song_name, _track_text, _player_state, _player_volume
    _saga_name = saga_name
    _song_name = song_name
    _track_text = track_text
    _player_state = player_state
    _player_volume = volume


def consume_pending_action() -> str:
    return _get_gesture().consume_pending_action()


def consume_pending_volume() -> float | None:
    return _get_volume().consume_pending_volume()


def _set_action(action: str) -> None:
    global _action_text, _action_until
    if action:
        _action_text = action
        _action_until = time.monotonic() + MUSIC_ACTION_DISPLAY_SECONDS


def _current_action() -> str:
    return _action_text if time.monotonic() < _action_until else ""


def _draw_fingers_box(frame: NDArray[np.uint8], text: str | int) -> None:
    _put_text_box(
        frame,
        MUSIC_FINGERS_FMT.format(text),
        (10, 196),
        0.5,
        1,
        WHITE,
        BLACK,
        pad_x=5,
        pad_y=3,
    )


def _draw_action_line(frame: NDArray[np.uint8]) -> None:
    shown = _current_action()
    text = shown if shown else MUSIC_UNKNOWN
    _put_text_box(
        frame, MUSIC_ACTION_FMT.format(text), (10, 220), 0.5, 1, WHITE, BLACK, pad_x=5, pad_y=3
    )


def _bbox_tuple(pts: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _effective_zone(w: int, h: int) -> tuple[int, int, int, int]:
    margin_x = int(w * MUSIC_EFFECTIVE_ZONE_FRACTION)
    margin_y = int(h * MUSIC_EFFECTIVE_ZONE_FRACTION)
    return margin_x, margin_y, w - margin_x, h - margin_y


def _hand_center(pts: list[tuple[int, int]]) -> tuple[int, int]:
    x1, y1, x2, y2 = _bbox_tuple(pts)
    return (x1 + x2) // 2, (y1 + y2) // 2


def _point_in_zone(pt: tuple[int, int], zone: tuple[int, int, int, int]) -> bool:
    x, y = pt
    x1, y1, x2, y2 = zone
    return x1 <= x <= x2 and y1 <= y <= y2


def _draw_zone(frame: NDArray[np.uint8], zone: tuple[int, int, int, int], active: bool) -> None:
    x1, y1, x2, y2 = zone
    color = GREEN if active else GRAY
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    (tw, _), _ = cv2.getTextSize(MUSIC_ZONE_TITLE, FONT, 0.45, 1)
    cx = (x1 + x2) // 2
    _put_text_box(
        frame,
        MUSIC_ZONE_TITLE,
        (cx - tw // 2, y1 + 14),
        0.45,
        1,
        color,
        BLACK,
        pad_x=5,
        pad_y=3,
    )


def _draw_volume_ui(
    frame: NDArray[np.uint8],
    preview: float,
    bbox: tuple[int, int, int, int],
) -> None:
    x1, y1, x2, y2 = bbox
    text = MUSIC_VOLUME_FMT.format(round(preview * 100))
    org = (max(0, x1), max(18, y2 + 8))
    _put_text_box(frame, text, org, 0.5, 1, WHITE, BLACK, pad_x=5, pad_y=3)
    bar_w = 80
    bar_h = 6
    bx, by = max(0, x1), max(18, y2 + 30)
    cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), BLACK, -1)
    cv2.rectangle(frame, (bx, by), (bx + int(bar_w * preview), by + bar_h), WHITE, -1)


def draw(frame: NDArray[np.uint8], results) -> tuple[NDArray[np.uint8], int]:
    box = _put_text_box
    box(frame, spaced(MUSIC_TITLE), (10, 22), 0.70, 2, WHITE, BLACK)
    box(frame, MUSIC_SAGA_FMT.format(_saga_name), (10, 70), 0.55, 2, WHITE, BLACK, pad_x=5, pad_y=3)
    box(
        frame, MUSIC_SONG_FMT.format(_song_name), (10, 100), 0.45, 1, WHITE, BLACK, pad_x=5, pad_y=3
    )
    box(
        frame,
        MUSIC_COUNT_FMT.format(_track_text),
        (10, 124),
        0.45,
        1,
        WHITE,
        BLACK,
        pad_x=5,
        pad_y=3,
    )
    box(
        frame,
        MUSIC_STATE_FMT.format(_player_state),
        (10, 148),
        0.55,
        1,
        WHITE,
        BLACK,
        pad_x=5,
        pad_y=3,
    )
    vol_text = MUSIC_VOLUME_FMT.format(round(_player_volume * 100))
    box(frame, vol_text, (10, 172), 0.55, 1, WHITE, BLACK, pad_x=5, pad_y=3)
    h, w = frame.shape[:2]
    zone = _effective_zone(w, h)
    zone_hit = False
    _draw_zone(frame, zone, False)
    hand_count = count_hands(results)
    landmarks = results.hand_landmarks or []
    if hand_count == 0:
        draw_viewfinder_crosshair(frame, WHITE)
        reset_state()
        _draw_fingers_box(frame, MUSIC_UNKNOWN)
        _draw_action_line(frame)
        return frame, hand_count

    right_idx = next(
        (i for i in range(len(landmarks)) if get_handedness(results, i) == RIGHT),
        None,
    )
    left_idx = next(
        (i for i in range(len(landmarks)) if get_handedness(results, i) == LEFT),
        None,
    )
    if right_idx is None and left_idx is None:
        reset_state()
        _draw_fingers_box(frame, MUSIC_UNKNOWN)
        _draw_action_line(frame)
        return frame, hand_count

    if left_idx is not None:
        left_lm = landmarks[left_idx]
        left_pts = to_pixel_points(left_lm, w, h)
        draw_bbox(frame, left_pts, WHITE, thickness=2)
        draw_skeleton(frame, left_pts)
        if _point_in_zone(_hand_center(left_pts), zone):
            zone_hit = True
            vc = _get_volume()
            scale = hand_scale(left_lm)
            distance = index_thumb_distance(left_lm, scale)
            preview, _ = vc.feed(distance)
            _draw_volume_ui(frame, preview, _bbox_tuple(left_pts))
        lx = min(p[0] for p in left_pts)
        ly = min(p[1] for p in left_pts)
        org_l = (max(0, lx), max(18, ly - 30))
        _put_text_box(frame, MUSIC_HAND_L, org_l, 0.6, 2, WHITE, BLACK, pad_x=5, pad_y=3)

    if right_idx is not None:
        hand_landmarks = landmarks[right_idx]
        pts = to_pixel_points(hand_landmarks, w, h)
        draw_bbox(frame, pts, WHITE, thickness=2)
        draw_skeleton(frame, pts)
        if _point_in_zone(_hand_center(pts), zone):
            zone_hit = True
            feat = features_from_landmarks(hand_landmarks, w, h)
            count, action = _get_gesture().feed(feat)
            _set_action(action)
        else:
            _get_gesture().reset()
            count, action = -1, ""
        rx = min(p[0] for p in pts) - 10
        ry = min(p[1] for p in pts) - 10
        org_r = (max(0, rx), max(18, ry - 10))
        _put_text_box(frame, MUSIC_HAND_R, org_r, 0.6, 2, WHITE, BLACK, pad_x=5, pad_y=3)
    else:
        _get_gesture().reset()
        count, action = -1, ""

    _draw_fingers_box(frame, count if count >= 0 else MUSIC_UNKNOWN)
    _draw_action_line(frame)
    _draw_zone(frame, zone, zone_hit)

    return frame, hand_count
