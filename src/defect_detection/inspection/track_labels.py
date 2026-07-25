"""per-track class-label stabilization via confidence-weighted voting.

one physical bottle has one true state, so a track's label commits once
enough evidence accumulates and never flips afterwards (deepstream-style
id-keyed decision cache). uncommitted tracks expose their current best
label so centerline decisions are never blocked.

evidence quality is spatially correlated with the centerline decision zone:
the model is only reliable when the bottle is fully visible near the line, so
votes are accepted only from detections whose centroid falls inside an
evidence zone bracketing the centerline. approach/exit frames (small, occluded,
off-center bottles) are excluded by design — this is the structural fix for
commit-on-approach-noise observed in video5 replay (tracks 9 & 15).
"""

from __future__ import annotations

from collections import defaultdict

from defect_detection.inspection.types import Detection

UNKNOWN = "unknown"


class TrackLabelStabilizer:
    """confidence-weighted per-track class label with one-shot commitment.

    three knobs:
      commit_score: cumulative confidence threshold to freeze a track's label.
      min_frames:   minimum in-zone observations before commitment is allowed.
      zone_frac:    fraction of frame width on EACH side of center that counts
                    as the evidence zone (abs(cx - mid_x) <= zone_frac*width).
    """

    def __init__(
        self,
        commit_score: float = 3.0,
        min_frames: int = 2,
        zone_frac: float = 0.15,
    ):
        self.commit_score = commit_score
        self.min_frames = min_frames
        self.zone_frac = zone_frac
        # track_id -> {class: cumulative_confidence}
        self.votes: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        # track_id -> number of in-zone usable (non-unknown, tracked) observations
        self.frames_seen: dict[int, int] = defaultdict(int)
        # track_id -> frozen label; once set, never changes
        self.committed: dict[int, str] = {}

    def update(self, detections: list[Detection], frame_width: int) -> None:
        """accumulate evidence for in-zone, tracked, classified detections.

        a detection votes only if its centroid is within the centerline
        evidence zone (abs(cx - mid_x) <= zone_frac * frame_width). this
        matches the system's design invariant: decisions are made at the line,
        and only frames near the line carry reliable class evidence.

        already-committed tracks are skipped (frozen forever). unknown class
        and track_id=None entries are ignored entirely.
        """
        mid_x = frame_width // 2
        zone_half = self.zone_frac * frame_width
        for det in detections:
            track_id = det.get("track_id")
            cls = det.get("defect_type")
            if track_id is None or cls is None or cls == UNKNOWN:
                continue
            if track_id in self.committed:
                # ponytail: frozen — deepstream-style id-keyed cache; upgrade =
                # an explicit "major evidence" override (e.g. long sustained
                # opposing in-zone run) if a track ever legitimately needs to flip.
                continue
            x, _y, w, _h = det["bbox"]
            cx = x + w // 2
            # ponytail: in-zone gate — evidence quality is spatially correlated
            # with the decision zone; off-center approach/exit frames are excluded
            # by design. ceiling: calibrated on a single video; upgrade path =
            # per-class confidence calibration, a learned zone, or
            # track-velocity/size-weighted evidence rather than a fixed band.
            if abs(cx - mid_x) > zone_half:
                continue
            conf = float(det.get("confidence", 0.0))
            self.votes[track_id][cls] += conf
            self.frames_seen[track_id] += 1
            scores = self.votes[track_id]
            if self.frames_seen[track_id] >= self.min_frames and scores:
                top = max(scores, key=scores.get)
                if scores[top] >= self.commit_score:
                    self.committed[track_id] = top

    def label_for(self, track_id: int) -> str | None:
        """committed label if present; else current argmax of votes; None if no votes."""
        if track_id in self.committed:
            return self.committed[track_id]
        v = self.votes.get(track_id)
        if not v:
            return None
        # ponytail: plain argmax ties broken by insertion order (dict stability);
        # insertion order = first-seen in-zone class wins, which matches "one
        # bottle, one state" intuition. upgrade = a proper prior or bard-score tiebreak.
        return max(v, key=v.get)

    def reset(self) -> None:
        """clear all state (track ids restart → vote state is meaningless)."""
        self.votes.clear()
        self.frames_seen.clear()
        self.committed.clear()
