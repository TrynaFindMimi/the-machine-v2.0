from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from config.settings import CAMERA_INDEX


class Camera:
    def __init__(self, w: int, h: int, device: int = CAMERA_INDEX) -> None:
        if w <= 0 or h <= 0:
            raise ValueError("dimensiones de cámara deben ser > 0")
        self.w: int = int(w)
        self.h: int = int(h)
        self.device: int = device
        self._vc = cv2.VideoCapture(device)
        if not self._vc.isOpened():
            raise RuntimeError(
                f"no se pudo abrir la cámara (índice={device}). "
                "Verifica que esté conectada y, si es otra cámara, "
                "ajusta THE_MACHINE_CAMERA_INDEX (ej. 1)."
            )
        self._vc.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self._vc.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    def read(self) -> NDArray[np.uint8] | None:
        ret, frame = self._vc.read()
        if not ret or frame is None:
            return None
        frame = cv2.flip(frame, 1)
        if frame.shape[1] != self.w or frame.shape[0] != self.h:
            frame = cv2.resize(frame, (self.w, self.h))
        return frame.astype(np.uint8)

    def release(self) -> None:
        if self._vc is not None:
            self._vc.release()

    def is_opened(self) -> bool:
        return self._vc.isOpened()

    def __enter__(self) -> Camera:
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()

    @property
    def vc(self) -> cv2.VideoCapture:
        return self._vc
