"""video capture helper shared by cli and gui"""

from collections.abc import Callable

import cv2
import numpy as np


class FrameReader:
    """reads frames from a video source, optionally looping at eof"""

    def __init__(
        self,
        source: str | int,
        loop: bool = False,
        on_loop: Callable[[], None] | None = None,
    ):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open video source: {source}")
        self.loop = loop
        self.on_loop = on_loop

    def read(self) -> np.ndarray | None:
        """next frame, or None if the source is exhausted (loop=False) or unreadable"""
        ret, frame = self.cap.read()
        if not ret and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if self.on_loop:
                self.on_loop()
            ret, frame = self.cap.read()
        return frame if ret else None

    def release(self) -> None:
        self.cap.release()
