"""detection dict type"""

from typing import TypedDict


class Detection(TypedDict, total=False):
    bbox: tuple[int, int, int, int]
    confidence: float
    class_id: int
    defect_type: str
    track_id: int | None
    bottle_id: str
    display_id: str
    on_centerline: bool
    logged: bool
