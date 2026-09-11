from __future__ import annotations

import time


class FPSCounter:
    def __init__(self, smoothing: float = 0.9) -> None:
        if not 0.0 <= smoothing <= 1.0:
            raise ValueError("smoothing debe estar en [0,1]")
        self.smoothing: float = smoothing
        self._last: float = time.perf_counter()
        self.fps: float = 0.0

    def tick(self) -> float:
        now = time.perf_counter()
        dt = now - self._last
        self._last = now
        if dt > 0:
            instant = 1.0 / dt
            if self.fps == 0.0:
                self.fps = instant
            else:
                self.fps = self.smoothing * self.fps + (1 - self.smoothing) * instant
        return self.fps

    def reset(self) -> None:
        self._last = time.perf_counter()
        self.fps = 0.0
