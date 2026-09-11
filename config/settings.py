from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Final


def _env_audio_driver() -> str:
    return os.environ.get("THE_MACHINE_AUDIO_DRIVER", "wasapi" if os.name == "nt" else "pulseaudio")


def _env_audio_device() -> str | None:
    return os.environ.get("THE_MACHINE_AUDIO_DEVICE") or None


def _env_camera_index() -> int:
    try:
        return int(os.environ.get("THE_MACHINE_CAMERA_INDEX", "0"))
    except ValueError:
        return 0


@dataclass(frozen=True, slots=True)
class WindowSettings:
    width: int = 1280
    height: int = 720
    sidebar_width: int = 220
    title: str = "the-machine"
    target_fps: int = 30

    @property
    def camera_width(self) -> int:
        return self.width - self.sidebar_width

    @property
    def camera_height(self) -> int:
        return self.height


@dataclass(frozen=True, slots=True)
class VisionSettings:
    landmarker_path: str = "models/hand_landmarker.task"
    gesture_path: str = "models/gesture_recognizer.task"
    num_hands_landmarker: int = 2
    num_hands_gesture: int = 2
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    inference_scale: float = 0.6


@dataclass(frozen=True, slots=True)
class AudioSettings:
    driver: str = field(default_factory=_env_audio_driver)
    device: str | None = field(default_factory=_env_audio_device)
    frequency: int = 44100
    size: int = -16
    channels: int = 2
    buffer: int = 512
    volume: float = 1.0


@dataclass(frozen=True, slots=True)
class PerceptronSettings:
    n_samples: int = 24
    margin: float = 0.03
    learning_rate: float = 0.05
    max_epochs: int = 3000
    fingers_together_thresh: float = 0.04
    epoch_budget: int = 200
    point_a_idx: int = 4
    point_b_idx: int = 8


@dataclass(frozen=True, slots=True)
class ThumbSettings:
    fold_dist: float = 0.10
    extend_dist: float = 0.13
    model_path: str = "models/thumb.npz"


@dataclass(frozen=True, slots=True)
class VolumeSettings:
    near: float = 0.10
    far: float = 0.80
    smoothing: float = 0.35
    model_path: str = "models/volume.npz"


MUSIC_ACTION_DISPLAY_SECONDS: Final = 1.0
MUSIC_EFFECTIVE_ZONE_FRACTION: Final = 1 / 8


WINDOW: Final = WindowSettings()
VISION: Final = VisionSettings()
AUDIO: Final = AudioSettings()
PERCEPTRON: Final = PerceptronSettings()
VOLUME: Final = VolumeSettings()
THUMB: Final = ThumbSettings()
CAMERA_INDEX: Final = _env_camera_index()
