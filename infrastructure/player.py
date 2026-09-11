from __future__ import annotations

import os
import pathlib
from typing import Final

import numpy as np
import pygame

from config.settings import AUDIO
from config.strings import MUSIC_STATE_PAUSE, MUSIC_STATE_PLAY, MUSIC_UNKNOWN
from core.playlist import scan_sagas

AUDIO_DIR: Final = pathlib.Path("music")


class MusicPlayer:
    def __init__(self, base: pathlib.Path = AUDIO_DIR) -> None:
        self.base = base
        self.sagas: list[list[pathlib.Path]] = scan_sagas(base)
        self.saga_idx: int = 0
        self.song_idx: int = 0
        self.playing: bool = False
        self.paused: bool = False
        self.volume: float = float(AUDIO.volume)
        self.audio_ok: bool = True
        self._loaded: pathlib.Path | None = None

    @property
    def current_saga(self) -> list[pathlib.Path] | None:
        if not self.sagas:
            return None
        return self.sagas[self.saga_idx % len(self.sagas)]

    @property
    def current_song(self) -> pathlib.Path | None:
        saga = self.current_saga
        if not saga:
            return None
        return saga[self.song_idx % len(saga)]

    @property
    def current_saga_name(self) -> str:
        if not self.sagas:
            return MUSIC_UNKNOWN
        return self.sagas[self.saga_idx % len(self.sagas)][0].parent.name

    def _init_mixer(self) -> None:
        def init(dev: str | None) -> None:
            pygame.mixer.init(
                frequency=AUDIO.frequency,
                size=AUDIO.size,
                channels=AUDIO.channels,
                buffer=AUDIO.buffer,
                devicename=dev,
            )

        for driver in (AUDIO.driver, None):
            if driver is None:
                os.environ.pop("SDL_AUDIODRIVER", None)
            else:
                os.environ["SDL_AUDIODRIVER"] = driver
            if pygame.mixer.get_init():
                pygame.mixer.quit()
            try:
                init(AUDIO.device or None)
                self.audio_ok = True
                return
            except pygame.error:
                continue
        pygame.mixer.quit()
        os.environ.pop("SDL_AUDIODRIVER", None)
        try:
            pygame.mixer.init()
            self.audio_ok = True
        except pygame.error as e:
            self.audio_ok = False
            print(f"[player] sin audio disponible, reproducción desactivada: {e}")

    def play(self) -> None:
        song = self.current_song
        if song is None:
            return
        if not self.audio_ok:
            return
        if not pygame.mixer.get_init():
            self._init_mixer()
        if not self.audio_ok:
            return
        if self._loaded == song and self.paused:
            pygame.mixer.music.unpause()
            self.paused = False
            self.playing = True
            return
        if self._loaded == song and self.playing:
            return
        try:
            pygame.mixer.music.load(str(song))
            pygame.mixer.music.set_volume(self.volume)
            pygame.mixer.music.play()
        except Exception as e:
            print(f"[player] error cargando {song}: {e}")
            return
        self._loaded = song
        self.paused = False
        self.playing = True

    def pause(self) -> None:
        if self.playing and self.audio_ok:
            pygame.mixer.music.pause()
            self.paused = True
            self.playing = False

    def resume(self) -> None:
        if self.paused and self.audio_ok:
            pygame.mixer.music.unpause()
            self.paused = False
            self.playing = True

    def stop(self) -> None:
        if self.audio_ok and pygame.mixer.get_init():
            pygame.mixer.music.stop()
        self.playing = False
        self.paused = False
        self._loaded = None
        self.saga_idx = 0
        self.song_idx = 0

    def set_volume(self, volume: float) -> None:
        self.volume = float(np.clip(volume, 0.0, 1.0))
        if pygame.mixer.get_init():
            pygame.mixer.music.set_volume(self.volume)

    def next_song(self) -> None:
        saga = self.current_saga
        if not saga:
            return
        self.song_idx = (self.song_idx + 1) % len(saga)
        self.play()

    def prev_song(self) -> None:
        saga = self.current_saga
        if not saga:
            return
        self.song_idx = (self.song_idx - 1) % len(saga)
        self.play()

    def next_saga(self) -> None:
        if not self.sagas:
            return
        self.saga_idx = (self.saga_idx + 1) % len(self.sagas)
        self.song_idx = 0
        self.play()

    def prev_saga(self) -> None:
        if not self.sagas:
            return
        self.saga_idx = (self.saga_idx - 1) % len(self.sagas)
        self.song_idx = 0
        self.play()

    def state(self) -> str:
        return MUSIC_STATE_PLAY if self.playing else MUSIC_STATE_PAUSE

    def tick(self) -> None:
        if not self.playing or self.paused or not self.audio_ok:
            return
        if pygame.mixer.get_init() and not pygame.mixer.music.get_busy():
            self.next_song()
