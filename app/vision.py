from __future__ import annotations

import pathlib

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    GestureRecognizer,
    GestureRecognizerOptions,
    HandLandmarker,
    HandLandmarkerOptions,
)
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

from config.settings import VISION, VisionSettings

LANDMARKER_PATH: str = VISION.landmarker_path
GESTURE_PATH: str = VISION.gesture_path


def _ensure_model(path: str) -> str:
    if not pathlib.Path(path).exists():
        raise FileNotFoundError(f"modelo no encontrado: {path}")
    return path


def make_landmarker(settings: VisionSettings = VISION) -> HandLandmarker:
    _ensure_model(settings.landmarker_path)
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=settings.landmarker_path),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_hands=settings.num_hands_landmarker,
        min_hand_detection_confidence=settings.min_detection_confidence,
        min_tracking_confidence=settings.min_tracking_confidence,
    )
    return HandLandmarker.create_from_options(options)


def make_recognizer(settings: VisionSettings = VISION) -> GestureRecognizer:
    _ensure_model(settings.gesture_path)
    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=settings.gesture_path),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_hands=settings.num_hands_gesture,
        min_hand_detection_confidence=settings.min_detection_confidence,
        min_tracking_confidence=settings.min_tracking_confidence,
    )
    return GestureRecognizer.create_from_options(options)
