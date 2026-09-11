from __future__ import annotations

import time
from typing import Final

import numpy as np
from numpy.typing import NDArray

from controllers.hand import HandController
from controllers.thumb import ThumbPerceptron

WINDOW_LEN: Final = 8
AGREE_REQUIRED: Final = 4
RATE_LIMIT_SECONDS: Final = 1.5
SETTLE_SECONDS: Final = 0.45


class MusicGestureController:
    def __init__(self, window: int = WINDOW_LEN) -> None:
        self.window_len = window
        self.hand = HandController()
        self.thumb = ThumbPerceptron()
        self.window: list[NDArray[np.float64]] = []
        self.agreement: int = 0
        self.last_count: int = -1
        self._last_change_at: float = 0.0
        self._last_fired: int = -1
        self._cooldown_until: float = 0.0
        self.pending: str = ""

    def load_or_train(self) -> None:
        self.thumb.load_or_train()
        if not self.hand.load():
            self.hand = HandController.train_all(window=self.window_len)

    def reset(self) -> None:
        self.window = []
        self.agreement = 0
        self.last_count = -1
        self._last_change_at = time.monotonic()
        self._last_fired = -1
        self._cooldown_until = 0.0
        self.pending = ""

    def consume_pending_action(self) -> str:
        action = self.pending
        self.pending = ""
        return action

    def _apply_action(self, count: int) -> str:
        now = time.monotonic()
        if count == self._last_fired:
            return ""
        if now < self._cooldown_until:
            return ""
        self._cooldown_until = now + RATE_LIMIT_SECONDS
        self._last_fired = count
        action = self.hand.action(count)
        if action:
            self.pending = action
        return action

    def feed(self, feat: NDArray[np.float64]) -> tuple[int, str]:
        self.window.append(feat)
        if len(self.window) > self.window_len:
            self.window = self.window[-self.window_len:]
        if len(self.window) < self.window_len:
            return -1, ""
        window_np = np.stack(self.window, axis=0)
        count = self.hand.count(window_np)
        now = time.monotonic()
        if count == self.last_count:
            self.agreement += 1
        else:
            self.agreement = 1
            self.last_count = count
            self._last_change_at = now
            self._last_fired = -1
        if (
            self.agreement >= AGREE_REQUIRED
            and now - self._last_change_at >= SETTLE_SECONDS
        ):
            return count, self._apply_action(count)
        return count, ""