from __future__ import annotations

import pathlib
from types import SimpleNamespace
from typing import Final

import numpy as np

from config.settings import THUMB
from core.fingers import (
    THUMB_PALM_IDS,
    hand_scale,
    set_thumb_weights,
    thumb_palm_features,
    thumb_tip_inside,
)
from core.perceptron import Perceptron

_THUMB_SEED: Final = 13


def _landmarks(p4: tuple[float, float]):
    ids = {
        0: (0.50, 0.80),
        1: (0.36, 0.70),
        2: (0.335, 0.660),
        3: (0.32, 0.620),
        4: p4,
        5: (0.42, 0.600),
        6: (0.42, 0.470),
        7: (0.42, 0.520),
        8: (0.42, 0.400),
        9: (0.48, 0.585),
        10: (0.48, 0.470),
        11: (0.48, 0.450),
        12: (0.48, 0.420),
        13: (0.54, 0.595),
        14: (0.54, 0.490),
        15: (0.54, 0.470),
        16: (0.54, 0.440),
        17: (0.61, 0.630),
        18: (0.61, 0.520),
        19: (0.61, 0.500),
        20: (0.61, 0.470),
    }
    pts = [SimpleNamespace(x=0.0, y=0.0) for _ in range(21)]
    for idx, (x, y) in ids.items():
        pts[idx] = SimpleNamespace(x=x, y=y)
    return pts


def _min_dist_palm(x: float, y: float) -> float:
    pts = _landmarks((0.45, 0.60))
    return min(float(((x - pts[i].x) ** 2 + (y - pts[i].y) ** 2) ** 0.5) for i in THUMB_PALM_IDS)


def _build_dataset(n: int = 400, seed: int = _THUMB_SEED):
    rng = np.random.default_rng(seed)
    xs: list[list[float]] = []
    ys: list[int] = []
    while len(ys) < n:
        x = rng.uniform(0.36, 0.62)
        y = rng.uniform(0.42, 0.75)
        lm = _landmarks((x, y))
        if not (thumb_tip_inside(lm) or _min_dist_palm(x, y) < THUMB.fold_dist):
            continue
        xs.append(thumb_palm_features(lm, hand_scale(lm)).tolist())
        ys.append(-1)
    while len(ys) < 2 * n:
        x = rng.uniform(0.14, 0.34)
        y = rng.uniform(0.30, 0.56)
        lm = _landmarks((x, y))
        if thumb_tip_inside(lm) or _min_dist_palm(x, y) < THUMB.extend_dist:
            continue
        xs.append(thumb_palm_features(lm, hand_scale(lm)).tolist())
        ys.append(1)
    xa = np.asarray(xs, dtype=np.float64)
    ya = np.asarray(ys, dtype=np.int64)
    p = rng.permutation(len(ya))
    return xa[p], ya[p]


class ThumbPerceptron:
    def __init__(self) -> None:
        self.perceptron = Perceptron(n_features=3)

    def load_or_train(self) -> None:
        path = pathlib.Path(THUMB.model_path)
        if path.is_file():
            data = np.load(path)
            if "w" in data and data["w"].shape == (3,):
                self.perceptron.w = data["w"]
                set_thumb_weights(self.perceptron.w)
                return
        self._train()
        self.save()
        set_thumb_weights(self.perceptron.w)

    def _train(self) -> None:
        X, y = _build_dataset()
        self.perceptron.reset()
        self.perceptron.train(X, y)
        acc = sum(1 for xi, yi in zip(X, y) if self.perceptron.predict(xi) == int(yi))
        print(f"[thumb] perceptron acc={acc / len(y):.3f}")

    def save(self) -> None:
        path = pathlib.Path(THUMB.model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, w=self.perceptron.w)

    def decision(self, hand_landmarks) -> bool:
        if self.perceptron.w is None or not np.any(self.perceptron.w):
            return False
        feat = thumb_palm_features(hand_landmarks, hand_scale(hand_landmarks))
        return self.perceptron.predict(feat) == 1
