from __future__ import annotations

from typing import Final

from config.strings import (
    GESTURE_CLOSED_FIST,
    GESTURE_I_LOVE_YOU,
    GESTURE_NONE,
    GESTURE_OPEN_PALM,
    GESTURE_POINTING_UP,
    GESTURE_THUMB_DOWN,
    GESTURE_THUMB_UP,
    GESTURE_VICTORY,
)

WHITE: Final = (245, 245, 245)
BLACK: Final = (0, 0, 0)
GREEN: Final = (0, 255, 0)
CYAN: Final = (0, 255, 255)
RED: Final = (0, 0, 255)
BLUE: Final = (255, 60, 20)

GRAY: Final = (140, 140, 140)

SIDEBAR_BG: Final = (18, 18, 18)
SIDEBAR_BORDER: Final = (70, 70, 70)

# --- Paleta griega (EPIC: The Musical) ---
# marfil (marmol) texto principal
IVORY: Final = (227, 236, 240)
# piedra clara atenuada
STONE: Final = (146, 158, 168)
# basalto oscuro (fondo templo)
BASALT: Final = (26, 30, 40)
# bronce (relieves, bordes)
BRONZE: Final = (70, 108, 160)
# dorado (acentos)
GOLD: Final = (96, 178, 210)
# verde oliva (laurel / live)
OLIVE: Final = (100, 142, 122)
# terracota (gestos/accion)
CLAY: Final = (74, 96, 190)
# marmol claro para paneles
MARBLE: Final = (208, 218, 226)

GESTURE_COLORS: Final[dict[str, tuple[int, int, int]]] = {
    GESTURE_NONE: RED,
    GESTURE_CLOSED_FIST: (0, 140, 255),
    GESTURE_OPEN_PALM: GREEN,
    GESTURE_POINTING_UP: CYAN,
    GESTURE_THUMB_DOWN: (0, 0, 255),
    GESTURE_THUMB_UP: (0, 255, 100),
    GESTURE_VICTORY: (255, 200, 0),
    GESTURE_I_LOVE_YOU: (200, 100, 255),
}
