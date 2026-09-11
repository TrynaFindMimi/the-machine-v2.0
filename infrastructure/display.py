from __future__ import annotations

import cv2
import numpy as np
import pygame
from numpy.typing import NDArray


class Window:
    def __init__(self, w: int, h: int, title: str) -> None:
        if w <= 0 or h <= 0:
            raise ValueError("dimensiones de ventana deben ser > 0")
        self.w: int = int(w)
        self.h: int = int(h)
        self.title: str = title
        if not pygame.get_init():
            pygame.init()
        pygame.display.set_caption(title)
        self.screen = pygame.display.set_mode((w, h))

    def show(self, canvas: NDArray[np.uint8]) -> None:
        rgb_out = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        surface = pygame.image.frombuffer(rgb_out.tobytes(), (self.w, self.h), "RGB")
        self.screen.blit(surface, (0, 0))
        pygame.display.flip()

    def poll(self) -> list[pygame.event.Event]:
        return pygame.event.get()

    def close(self) -> None:
        pygame.quit()

    def __enter__(self) -> Window:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
