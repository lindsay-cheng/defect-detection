"""centerline counting, display ids, and session state (pure logic)"""

from dataclasses import dataclass, field
from datetime import datetime

from defect_detection.constants import DEFECT_TYPE_GOOD
from defect_detection.inspection.track_labels import TrackLabelStabilizer
from defect_detection.inspection.types import Detection


@dataclass
class InspectionResult:
    """per-frame output of Inspector.process: detections needing side effects"""

    counted: list[Detection] = field(default_factory=list)
    defects: list[Detection] = field(default_factory=list)


class Inspector:
    """centerline counting, display ids, and session state (pure logic)"""

    def __init__(self, centerline_tolerance: int = 15):
        self.centerline_tolerance = centerline_tolerance

        # stats
        self.total_inspected = 0
        self.total_defects = 0
        # dedupe sets keyed by track_id (int)
        self.counted_tracks: set[int] = set()
        self.logged_tracks: set[int] = set()

        # operator-facing consecutive numbering; reset each session
        self.session_id: str = ""
        self.next_display_number: int = 1
        self.display_number_by_track_id: dict[int, int] = {}

        # per-track class-label stabilization (deepstream-style id-keyed cache)
        self._labels = TrackLabelStabilizer()

    def start_session(self) -> None:
        """begin a new inspection run — resets stats and display numbering"""
        self.counted_tracks.clear()
        self.logged_tracks.clear()
        self.display_number_by_track_id.clear()
        self.total_inspected = 0
        self.total_defects = 0
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.next_display_number = 1
        self._labels.reset()

    def reset_tracks(self) -> None:
        """reset track-keyed state so track ids restart.
        also clears all track-id-keyed state (dedupe sets, display mapping)
        since those are meaningless once the tracker reassigns ids from 1.
        session_id and next_display_number are preserved so display numbering
        continues uninterrupted across video loops within the same session.
        """
        self.counted_tracks.clear()
        self.logged_tracks.clear()
        self.display_number_by_track_id.clear()
        self._labels.reset()

    def process(self, detections: list[Detection], frame_width: int) -> InspectionResult:
        """apply centerline detection, display ids, counting, and defect logging.
        returns the detections needing side effects (no side effects performed here)."""
        result = InspectionResult()

        # stabilize per-track class labels before any centerline/count/log logic
        # so the label at the crossing instant is the committed (or current-best)
        # one, not the flickering per-frame class. raw_defect_type preserves the
        # instantaneous model class for diagnostics; all downstream logic reads
        # defect_type as before.
        self._labels.update(detections, frame_width)
        for det in detections:
            track_id = det.get("track_id")
            if track_id is None:
                # ponytail: no track -> no epistemic state worth committing; mark
                # evaluating so the UI renders neutral. upgrade = a transient-id
                # path for untracked but high-conf singletons.
                det["label_state"] = "evaluating"
                continue
            det["raw_defect_type"] = det.get("defect_type")
            stable = self._labels.label_for(track_id)
            if stable is not None:
                det["defect_type"] = stable
            det["label_state"] = (
                "committed" if self._labels.is_committed(track_id) else "evaluating"
            )

        mid_x = frame_width // 2
        for detection in detections:
            cx = detection["bbox"][0] + detection["bbox"][2] // 2
            detection["on_centerline"] = abs(cx - mid_x) <= self.centerline_tolerance

        self._assign_display_ids(detections)
        self._count_inspected(detections, result)
        self._log_detections(detections, result)

        return result

    def _assign_display_ids(self, detections: list[Detection]):
        """assign a consecutive operator-facing display_id on the first centerline hit per track"""
        for detection in detections:
            track_id = detection.get("track_id")
            if track_id is None:
                continue
            if track_id in self.display_number_by_track_id:
                n = self.display_number_by_track_id[track_id]
            elif detection.get("on_centerline"):
                n = self.next_display_number
                self.display_number_by_track_id[track_id] = n
                self.next_display_number += 1
            else:
                continue
            detection["display_id"] = f"BTL_{n:05d}"

    def _count_inspected(self, detections: list[Detection], result: InspectionResult):
        """count unique bottles on the vertical center line.
        defective bottles are later upserted to FAIL by _log_detections.
        uses the on_centerline flag computed once in process().
        """
        for detection in detections:
            if not detection.get("on_centerline"):
                continue
            track_id = detection.get("track_id")
            if track_id is None or track_id in self.counted_tracks:
                continue
            self.counted_tracks.add(track_id)
            self.total_inspected += 1
            result.counted.append(detection)

    def _log_detections(self, detections: list[Detection], result: InspectionResult):
        """log defective bottles when centroid is on the center line.
        uses the on_centerline flag computed once in process()."""
        for detection in detections:
            if not detection.get("on_centerline"):
                continue
            track_id = detection.get("track_id")
            defect_type = detection.get("defect_type")

            if track_id is None or defect_type == DEFECT_TYPE_GOOD:
                continue
            if track_id in self.logged_tracks:
                continue

            self.logged_tracks.add(track_id)
            self.total_defects += 1
            detection["logged"] = True
            result.defects.append(detection)
