from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class ModeHandler(Protocol):
    def __call__(self, frame: NDArray[np.uint8], results) -> tuple[NDArray[np.uint8], int]: ...
