from __future__ import annotations

import contextlib
import time
from typing import Final

import cv2
import mediapipe as mp
import numpy as np
import pygame

from app.registry import TEST_NAMES, TESTS
from app.vision import make_landmarker, make_recognizer
from config.settings import VISION, WINDOW, VisionSettings, WindowSettings
from config.strings import (
    MUSIC_ACTION_NEXT_SAGA,
    MUSIC_ACTION_NEXT_SONG,
    MUSIC_ACTION_PAUSE,
    MUSIC_ACTION_PLAY,
    MUSIC_ACTION_PREV_SAGA,
    MUSIC_ACTION_PREV_SONG,
    MUSIC_ACTION_STOP,
    MUSIC_UNKNOWN,
)
from infrastructure.capture import Camera
from infrastructure.display import Window
from infrastructure.player import MusicPlayer
from presentation.ui.effects import apply_cctv_effect
from presentation.ui.layout import draw_sidebar

WINDOW_W: Final = WINDOW.width
WINDOW_H: Final = WINDOW.height
CAMERA_W: Final = WINDOW.camera_width
CAMERA_H: Final = WINDOW.camera_height


def _handle_events(mode_idx: int) -> tuple[bool, int]:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True, mode_idx
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                return True, mode_idx
            if event.key == pygame.K_n:
                return False, (mode_idx + 1) % len(TEST_NAMES)
    return False, mode_idx


def run(
    mode_idx: int = 0,
    window_cfg: WindowSettings = WINDOW,
    vision_cfg: VisionSettings | None = None,
) -> None:
    if not 0 <= mode_idx < len(TEST_NAMES):
        raise ValueError(f"mode_idx fuera de rango: {mode_idx}")

    vision = vision_cfg or VISION
    landmarker = make_landmarker(vision)
    recognizer = make_recognizer(vision)
    cam = Camera(window_cfg.camera_width, window_cfg.camera_height)
    window = Window(window_cfg.width, window_cfg.height, window_cfg.title)
    clock = pygame.time.Clock()
    player = MusicPlayer()

    cam_w = window_cfg.camera_width
    cam_h = window_cfg.camera_height
    infer_w = max(160, int(cam_w * vision.inference_scale))
    infer_h = max(120, int(cam_h * vision.inference_scale))
    downscale = infer_w != cam_w
    canvas = np.zeros((window_cfg.height, window_cfg.width, 3), dtype=np.uint8)

    with contextlib.ExitStack() as stack:
        stack.callback(cam.release)
        stack.callback(landmarker.close)
        stack.callback(recognizer.close)
        stack.callback(pygame.quit)

        current = mode_idx
        while True:
            frame = cam.read()
            if frame is None:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            infer = (
                cv2.resize(rgb, (infer_w, infer_h), interpolation=cv2.INTER_AREA)
                if downscale
                else rgb
            )
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=infer)
            ts = int(time.time() * 1000)

            name = TEST_NAMES[current]
            if name == "position":
                results = recognizer.recognize_for_video(mp_img, ts)
            else:
                results = landmarker.detect_for_video(mp_img, ts)

            if name == "music":
                from presentation.modes import music as music_mode

                song = player.current_song
                saga = player.current_saga
                if saga and song:
                    track_text = f"{player.song_idx + 1}/{len(saga)}"
                else:
                    track_text = MUSIC_UNKNOWN
                music_mode.set_context(
                    player.current_saga_name,
                    song.name if song else MUSIC_UNKNOWN,
                    track_text,
                    player.state(),
                    player.volume,
                )

            out, hand_count = TESTS[name](frame, results)

            if name == "music":
                from presentation.modes import music as music_mode

                action = music_mode.consume_pending_action()
                if action == MUSIC_ACTION_PLAY:
                    player.play()
                elif action == MUSIC_ACTION_PAUSE:
                    player.pause()
                elif action == MUSIC_ACTION_STOP:
                    player.stop()
                elif action == MUSIC_ACTION_PREV_SONG:
                    player.prev_song()
                elif action == MUSIC_ACTION_NEXT_SONG:
                    player.next_song()
                elif action == MUSIC_ACTION_PREV_SAGA:
                    player.prev_saga()
                elif action == MUSIC_ACTION_NEXT_SAGA:
                    player.next_saga()
                volume = music_mode.consume_pending_volume()
                if volume is not None:
                    player.set_volume(volume)
                player.tick()
            fps = clock.get_fps()
            apply_cctv_effect(out)

            canvas.fill(0)
            canvas[:cam_h, :cam_w] = out
            draw_sidebar(canvas, name, hand_count, fps)
            window.show(canvas)

            should_quit, current = _handle_events(current)
            if should_quit:
                break
            clock.tick(window_cfg.target_fps)
