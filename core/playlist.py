from __future__ import annotations

import pathlib
from typing import Final

_AUDIO_EXTS: Final = (".mp3", ".ogg", ".wav")


def scan_sagas(base: pathlib.Path = pathlib.Path("music")) -> list[list[pathlib.Path]]:
    if not base.is_dir():
        return []
    sagas: list[list[pathlib.Path]] = []
    for saga_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        tracks = sorted(
            p for p in saga_dir.iterdir() if p.is_file() and p.suffix.lower() in _AUDIO_EXTS
        )
        if tracks:
            sagas.append(tracks)
    return sagas
