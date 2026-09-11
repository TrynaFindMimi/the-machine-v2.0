from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from config.settings import PERCEPTRON

N_SAMPLES: int = PERCEPTRON.n_samples
MARGIN: float = PERCEPTRON.margin
LR: float = PERCEPTRON.learning_rate
MAX_EPOCHS: int = PERCEPTRON.max_epochs
FINGERS_TOGETHER_THRESH: float = PERCEPTRON.fingers_together_thresh


class Perceptron:
    def __init__(self, lr: float = LR, max_epochs: int = MAX_EPOCHS, n_features: int = 3) -> None:
        if lr <= 0:
            raise ValueError("lr debe ser > 0")
        if max_epochs <= 0:
            raise ValueError("max_epochs debe ser > 0")
        if n_features <= 0:
            raise ValueError("n_features debe ser > 0")
        self.w: NDArray[np.float64] = np.zeros(n_features, dtype=np.float64)
        self.lr: float = float(lr)
        self.max_epochs: int = int(max_epochs)

    def predict(self, x: NDArray[np.float64] | list[float]) -> int:
        arr = np.asarray(x, dtype=float)
        return 1 if float(np.dot(self.w, arr)) >= 0 else -1

    def train(self, X: NDArray[np.float64], y: NDArray[np.int64]) -> int:
        for epoch in range(1, self.max_epochs + 1):
            errors = 0
            for xi, yi in zip(X, y):
                if self.predict(xi) != int(yi):
                    self.w += self.lr * int(yi) * np.asarray(xi, dtype=float)
                    errors += 1
            if errors == 0:
                return epoch
        return self.max_epochs

    def train_budget(self, X: NDArray[np.float64], y: NDArray[np.int64], budget: int) -> int:
        if budget <= 0:
            return 0
        trained = 0
        for _ in range(budget):
            errors = 0
            for xi, yi in zip(X, y):
                if self.predict(xi) != int(yi):
                    self.w += self.lr * int(yi) * np.asarray(xi, dtype=float)
                    errors += 1
            trained += 1
            if errors == 0:
                return trained
        return trained

    def reset(self) -> None:
        self.w = np.zeros(self.w.shape[0], dtype=np.float64)


def build_dataset(
    p5: NDArray[np.float64],
    p9: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.int64]] | tuple[None, None]:
    p5_arr = np.asarray(p5, dtype=float).reshape(2)
    p9_arr = np.asarray(p9, dtype=float).reshape(2)
    seg = p9_arr - p5_arr
    length = float(np.linalg.norm(seg))
    if length < 1e-6:
        return None, None
    perp = np.array([-seg[1], seg[0]], dtype=float) / length

    t = np.linspace(0.0, 1.0, N_SAMPLES).reshape(-1, 1)
    pos = p5_arr + t * seg
    neg = pos + MARGIN * perp

    X = np.hstack([np.vstack([pos, neg]), np.ones((len(pos) + len(neg), 1))])
    y = np.array([1] * len(pos) + [-1] * len(neg), dtype=np.int64)
    return X, y
