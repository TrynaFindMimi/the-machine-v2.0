from __future__ import annotations

from typing import Final

WINDOW: Final = "The Machine"
USAGE: Final = "uso: python main.py [{}]"

HAND_TITLE: Final = "deteccion de manos"
LINE_TITLE: Final = "deteccion de linea"
POSITION_TITLE: Final = "deteccion de gestos"
MUSIC_TITLE: Final = "music player"
MUSIC_STATE_PLAY: Final = "PLAY"
MUSIC_STATE_PAUSE: Final = "PAUSE"
MUSIC_ACTION_PLAY: Final = "PLAY"
MUSIC_ACTION_PAUSE: Final = "PAUSE"
MUSIC_ACTION_STOP: Final = "STOP + RESET"
MUSIC_ACTION_PREV_SONG: Final = "PREV SONG"
MUSIC_ACTION_NEXT_SONG: Final = "NEXT SONG"
MUSIC_ACTION_PREV_SAGA: Final = "PREV SAGA"
MUSIC_ACTION_NEXT_SAGA: Final = "NEXT SAGA"
MUSIC_SAGA_FMT: Final = "SAGA: {}"
MUSIC_SONG_FMT: Final = "SONG: {}"
MUSIC_COUNT_FMT: Final = "TRACK: {}"
MUSIC_STATE_FMT: Final = "STATE: {}"
MUSIC_FINGERS_FMT: Final = "FINGERS: {}"
MUSIC_ACTION_FMT: Final = "ACTION: {}"
MUSIC_UNKNOWN: Final = "-"
MUSIC_HAND_R: Final = "Right"
MUSIC_HAND_L: Final = "Left"
MUSIC_VOLUME_FMT: Final = "VOLUME: {:>3.0f}%"
MUSIC_ZONE_TITLE: Final = "ZONA EFECTIVA"
FPS_FMT: Final = "FPS {:05.1f}"

KEYBIND_HINT: Final = "[n]ext [q]uit"

MODES_ORDER: Final[tuple[str, ...]] = ("hand", "line", "position", "music")
MODE_LABELS: Final[dict[str, str]] = {
    "hand": HAND_TITLE,
    "line": LINE_TITLE,
    "position": POSITION_TITLE,
    "music": MUSIC_TITLE,
}

GESTURE_NONE: Final = "None"
GESTURE_CLOSED_FIST: Final = "Closed_Fist"
GESTURE_OPEN_PALM: Final = "Open_Palm"
GESTURE_POINTING_UP: Final = "Pointing_Up"
GESTURE_THUMB_DOWN: Final = "Thumb_Down"
GESTURE_THUMB_UP: Final = "Thumb_Up"
GESTURE_VICTORY: Final = "Victory"
GESTURE_I_LOVE_YOU: Final = "ILoveYou"

GESTURES: Final[tuple[str, ...]] = (
    GESTURE_NONE,
    GESTURE_CLOSED_FIST,
    GESTURE_OPEN_PALM,
    GESTURE_POINTING_UP,
    GESTURE_THUMB_DOWN,
    GESTURE_THUMB_UP,
    GESTURE_VICTORY,
    GESTURE_I_LOVE_YOU,
)
GESTURE_NAMES: Final = GESTURES
