from __future__ import annotations

import pathlib
from typing import Final

import numpy as np
from numpy.typing import NDArray

from config.strings import (
    MUSIC_ACTION_NEXT_SAGA,
    MUSIC_ACTION_NEXT_SONG,
    MUSIC_ACTION_PAUSE,
    MUSIC_ACTION_PLAY,
    MUSIC_ACTION_PREV_SAGA,
    MUSIC_ACTION_PREV_SONG,
)

_MODEL_DIR: Final = pathlib.Path("models")
_WINDOW: Final = 8


class FingerRNN:
    def __init__(
        self,
        hidden: int = 8,
        layers: int = 2,
        dropout: float = 0.3,
        lr: float = 0.08,
        seed: int = 0,
    ) -> None:
        self.hidden = hidden
        self.layers = layers
        self.dropout = dropout
        self.lr = lr
        rng = np.random.default_rng(seed)
        h = hidden
        s = 0.6 / np.sqrt(h)
        self.Wx: list[NDArray[np.float64]] = [rng.normal(0.0, s, (h, 1))]
        self.Wx += [rng.normal(0.0, s, (h, h)) for _ in range(layers - 1)]
        self.Wh: list[NDArray[np.float64]] = [
            rng.normal(0.0, s, (h, h)) for _ in range(layers)
        ]
        self.b: list[NDArray[np.float64]] = [np.zeros(h) for _ in range(layers)]
        self.Wy: NDArray[np.float64] = rng.normal(0.0, 0.5, (2, h))
        self.by: NDArray[np.float64] = np.zeros(2)

    def predict(self, seq: NDArray[np.float64]) -> int:
        _, ss = self._forward(seq, train=False)
        logits = self.Wy @ ss[-1][-1] + self.by
        return int(np.argmax(logits))

    def accuracy(self, xs: NDArray[np.float64], ys: NDArray[np.int64]) -> float:
        ok = sum(1 for xi, yi in zip(xs, ys) if self.predict(xi) == int(yi))
        return ok / max(1, len(xs))

    def train(self, xs: NDArray[np.float64], ys: NDArray[np.int64], epochs: int = 20) -> None:
        rng = np.random.default_rng(42)
        for _ in range(epochs):
            self._train_epoch(xs, ys, rng)

    def train_until(
        self,
        xs: NDArray[np.float64],
        ys: NDArray[np.int64],
        vxs: NDArray[np.float64],
        vys: NDArray[np.int64],
        target: float = 0.85,
        gap: float = 0.002,
        max_epochs: int = 200,
    ) -> tuple[int, float, float]:
        rng = np.random.default_rng(42)
        epochs = 0
        ta, va = 0.0, 0.0
        for epochs in range(1, max_epochs + 1):
            self._train_epoch(xs, ys, rng)
            ta = self.accuracy(xs, ys)
            va = self.accuracy(vxs, vys)
            if ta > target and va > target and abs(ta - va) <= gap:
                break
        return epochs, ta, va

    def _train_epoch(
        self, xs: NDArray[np.float64], ys: NDArray[np.int64], rng: np.random.Generator
    ) -> None:
        dWx: list[NDArray[np.float64]] = [np.zeros_like(w) for w in self.Wx]
        dWh: list[NDArray[np.float64]] = [np.zeros_like(w) for w in self.Wh]
        db: list[NDArray[np.float64]] = [np.zeros_like(x) for x in self.b]
        dWy = np.zeros_like(self.Wy)
        dby = np.zeros_like(self.by)

        for xi, yi in zip(xs, ys):
            T = xi.shape[0]
            masks = self._make_masks(rng, T)
            hs, ss = self._forward(xi, train=True, masks=masks)
            h_last = ss[-1][-1]
            logits = self.Wy @ h_last + self.by
            exp = np.exp(logits - logits.max())
            probs = exp / exp.sum()
            grad_y = probs.copy()
            grad_y[yi] -= 1.0
            dWy += np.outer(grad_y, h_last)
            dby += grad_y
            from_above = [np.zeros((T, self.hidden)) for _ in range(self.layers)]
            from_above[-1][T - 1] = self.Wy.T @ grad_y
            mask = (
                [np.ones((T, self.hidden)) for _ in range(self.layers)]
                if masks is None
                else masks
            )
            for l in range(self.layers - 1, -1, -1):
                acc_time = np.zeros(self.hidden)
                for t in range(T - 1, -1, -1):
                    delta = mask[l][t] * from_above[l][t] + acc_time
                    dz = delta * (1.0 - hs[l][t] * hs[l][t])
                    inp = xi[t] if l == 0 else ss[l - 1][t]
                    dWx[l] += np.outer(dz, inp)
                    dWh[l] += np.outer(
                        dz, hs[l][t - 1] if t > 0 else np.zeros(self.hidden)
                    )
                    db[l] += dz
                    acc_time = self.Wh[l].T @ dz
                    if l > 0:
                        from_above[l - 1][t] += self.Wx[l].T @ dz
        n = max(1, len(xs))
        for l in range(self.layers):
            self.Wx[l] -= self.lr * dWx[l] / n
            self.Wh[l] -= self.lr * dWh[l] / n
            self.b[l] -= self.lr * db[l] / n
        self.Wy -= self.lr * dWy / n
        self.by -= self.lr * dby / n

    def _make_masks(
        self, rng: np.random.Generator, T: int
    ) -> list[NDArray[np.float64]] | None:
        if self.dropout <= 0.0:
            return None
        keep = 1.0 - self.dropout
        return [
            (rng.random((T, self.hidden)) < keep).astype(np.float64) / keep
            for _ in range(self.layers)
        ]

    def _forward(
        self,
        xi: NDArray[np.float64],
        train: bool = False,
        masks: list[NDArray[np.float64]] | None = None,
    ) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
        T = xi.shape[0]
        hs: list[NDArray[np.float64]] = [
            np.zeros((T, self.hidden)) for _ in range(self.layers)
        ]
        ss: list[NDArray[np.float64]] = [
            np.zeros((T, self.hidden)) for _ in range(self.layers)
        ]
        h_prev = [np.zeros(self.hidden) for _ in range(self.layers)]
        for t in range(T):
            inp: NDArray[np.float64] = xi[t]
            for l in range(self.layers):
                h = np.tanh(self.Wx[l] @ inp + self.Wh[l] @ h_prev[l] + self.b[l])
                hs[l][t] = h
                if train and self.dropout > 0.0 and masks is not None:
                    s = h * masks[l][t]
                else:
                    s = h
                ss[l][t] = s
                h_prev[l] = h
                inp = s
        return hs, ss


class HandController:
    def __init__(self) -> None:
        self.fingers: list[FingerRNN] = [FingerRNN(seed=idx) for idx in range(4)]

    def count(self, window: NDArray[np.float64]) -> int:
        c = 1 if int(np.sum(window[:, 0] > 0.0) >= window.shape[0] // 2) else 0
        for i, rnn in enumerate(self.fingers):
            seq = window[:, i + 1 : i + 2]
            if rnn.predict(seq) == 1:
                c += 1
        return c

    def action(self, count: int) -> str:
        if count == 0:
            return MUSIC_ACTION_PREV_SAGA
        if count == 1:
            return MUSIC_ACTION_PREV_SONG
        if count == 2:
            return MUSIC_ACTION_PAUSE
        if count == 3:
            return MUSIC_ACTION_PLAY
        if count == 4:
            return MUSIC_ACTION_NEXT_SONG
        if count == 5:
            return MUSIC_ACTION_NEXT_SAGA
        return ""

    def save(self) -> None:
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        stale = _MODEL_DIR / "finger_0.npz"
        if stale.is_file():
            stale.unlink()
        for i, rnn in enumerate(self.fingers):
            d: dict[str, NDArray[np.float64]] = {"Wy": rnn.Wy, "by": rnn.by}
            for l in range(rnn.layers):
                d[f"Wx_{l}"] = rnn.Wx[l]
                d[f"Wh_{l}"] = rnn.Wh[l]
                d[f"b_{l}"] = rnn.b[l]
            np.savez(_MODEL_DIR / f"finger_{i + 1}.npz", **d)  # type: ignore[arg-type]

    def load(self) -> bool:
        ok = True
        for i, rnn in enumerate(self.fingers):
            p = _MODEL_DIR / f"finger_{i + 1}.npz"
            if not p.is_file():
                ok = False
                continue
            d = np.load(p)
            if "Wx_0" not in d or any(
                f"Wh_{l}" not in d or f"b_{l}" not in d
                for l in range(rnn.layers)
            ):
                ok = False
                continue
            for l in range(rnn.layers):
                rnn.Wx[l] = d[f"Wx_{l}"]
                rnn.Wh[l] = d[f"Wh_{l}"]
                rnn.b[l] = d[f"b_{l}"]
            rnn.Wy = d["Wy"]
            rnn.by = d["by"]
        return ok

    @staticmethod
    def train_all(
        window: int = _WINDOW, target: float = 0.85, gap: float = 0.002
    ) -> HandController:
        hc = HandController()
        rng = np.random.default_rng(7)
        names = ("index", "middle", "ring", "pinky")
        for i, rnn in enumerate(hc.fingers):
            xs: list[NDArray[np.float64]] = []
            ys: list[int] = []
            vxs: list[NDArray[np.float64]] = []
            vys: list[int] = []
            for p, n in ((0.00, 100), (0.20, 100), (0.42, 100)):
                for _ in range(n):
                    xs.append(HandController._noisy_window(rng, window, p))
                    ys.append(0)
            for p, n in ((0.65, 100), (0.82, 100), (1.00, 100)):
                for _ in range(n):
                    xs.append(HandController._noisy_window(rng, window, p))
                    ys.append(1)
            for p, n in ((0.00, 120), (0.20, 120), (0.42, 120)):
                for _ in range(n):
                    vxs.append(HandController._noisy_window(rng, window, p))
                    vys.append(0)
            for p, n in ((0.65, 120), (0.82, 120), (1.00, 120)):
                for _ in range(n):
                    vxs.append(HandController._noisy_window(rng, window, p))
                    vys.append(1)
            epochs, ta, va = rnn.train_until(
                np.stack(xs),
                np.asarray(ys, dtype=np.int64),
                np.stack(vxs),
                np.asarray(vys, dtype=np.int64),
                target=target,
                gap=gap,
            )
            print(
                f"[train_all] {names[i]:6s} epochs={epochs:3d} "
                f"acc={ta:.3f} val_acc={va:.3f}"
            )
        hc.save()
        return hc

    @staticmethod
    def _noisy_window(rng: np.random.Generator, window: int, p: float) -> NDArray[np.float64]:
        base = np.where(rng.uniform(0.0, 1.0, (window, 1)) < p, 1.0, -1.0)
        return (base + rng.normal(0.0, 0.12, base.shape)).astype(np.float64)