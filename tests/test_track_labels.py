"""tests for defect_detection.inspection.track_labels — per-track class-label
stabilization via zone-gated confidence-weighted voting, plus one Inspector
integration. evidence zone = abs(cx - mid_x) <= zone_frac*frame_width on each
side of center; only in-zone frames vote."""

from defect_detection.inspection.inspector import Inspector
from defect_detection.inspection.track_labels import TrackLabelStabilizer

FRAME_WIDTH = 200  # mid_x = 100


def _det(track_id, cls, conf, cx=100):
    """detection with centroid cx (bbox x = cx - w//2, w=2 -> x = cx-1)."""
    return {
        "bbox": (cx - 1, 0, 2, 10),
        "confidence": conf,
        "class_id": 0,
        "defect_type": cls,
        "track_id": track_id,
    }


class TestCommitOnce:
    def test_commits_after_enough_in_zone_frames_then_never_flips(self):
        s = TrackLabelStabilizer()  # commit_score=3.0, min_frames=2, zone_frac=0.15
        # 4 in-zone (cx=100, |100-100|=0 <= 30) no_cap@0.9 -> 3.6 >= 3.0, commit
        for _ in range(4):
            s.update([_det(1, "no_cap", 0.9, cx=100)], FRAME_WIDTH)
        assert s.label_for(1) == "no_cap"
        assert 1 in s.committed

        # 10 consecutive in-zone good@0.95 frames must NOT flip a committed track
        for _ in range(10):
            s.update([_det(1, "good", 0.95, cx=100)], FRAME_WIDTH)
        assert s.label_for(1) == "no_cap"
        assert s.committed[1] == "no_cap"


class TestUncommittedFallback:
    def test_single_in_zone_frame_returns_argmax_uncommitted(self):
        s = TrackLabelStabilizer()
        s.update([_det(1, "no_cap", 0.9, cx=100)], FRAME_WIDTH)
        # 0.9 < commit_score and frames_seen=1 < min_frames -> uncommitted
        assert 1 not in s.committed
        # but label_for still returns the current best (argmax)
        assert s.label_for(1) == "no_cap"


class TestConflictingEvidence:
    def test_alternating_in_zone_delays_commit_until_threshold_crossed(self):
        s = TrackLabelStabilizer()
        # alternate good/no_cap at 0.6 -> after 6 frames good=1.8, no_cap=1.8
        for _ in range(3):
            s.update([_det(1, "good", 0.6)], FRAME_WIDTH)
            s.update([_det(1, "no_cap", 0.6)], FRAME_WIDTH)
        assert 1 not in s.committed

        # push no_cap over the threshold: +3 frames -> total 3.6 >= 3.0
        for _ in range(3):
            s.update([_det(1, "no_cap", 0.6)], FRAME_WIDTH)
        assert s.committed[1] == "no_cap"
        assert s.label_for(1) == "no_cap"


class TestMinFramesGate:
    def test_single_high_conf_frame_does_not_commit(self):
        s = TrackLabelStabilizer()
        # 0.99 -> 0.99 < 3.0 AND frames_seen=1 < min_frames=2
        s.update([_det(1, "no_cap", 0.99)], FRAME_WIDTH)
        assert 1 not in s.committed
        assert s.label_for(1) == "no_cap"

    def test_two_high_conf_frames_still_below_score(self):
        s = TrackLabelStabilizer()
        # 2 frames at 0.99 -> 1.98 < 3.0, so uncommitted even past min_frames
        for _ in range(2):
            s.update([_det(1, "no_cap", 0.99)], FRAME_WIDTH)
        assert 1 not in s.committed
        assert s.label_for(1) == "no_cap"

    def test_four_high_conf_frames_commit(self):
        s = TrackLabelStabilizer()
        # 4 frames at 0.99 -> 3.96 >= 3.0 and frames_seen=4 >= 2 -> commit
        for _ in range(4):
            s.update([_det(1, "no_cap", 0.99)], FRAME_WIDTH)
        assert s.committed[1] == "no_cap"


class TestIgnoredEntries:
    def test_unknown_class_ignored(self):
        s = TrackLabelStabilizer()
        s.update([_det(1, "unknown", 0.99)], FRAME_WIDTH)
        assert s.label_for(1) is None
        assert 1 not in s.frames_seen

    def test_track_id_none_ignored(self):
        s = TrackLabelStabilizer()
        det = _det(1, "no_cap", 0.99)
        det["track_id"] = None
        s.update([det], FRAME_WIDTH)
        assert s.label_for(1) is None
        assert s.frames_seen == {}


class TestReset:
    def test_reset_clears_state(self):
        s = TrackLabelStabilizer()
        for _ in range(4):
            s.update([_det(1, "no_cap", 0.9)], FRAME_WIDTH)
        assert s.committed
        s.reset()
        assert s.votes == {}
        assert s.frames_seen == {}
        assert s.committed == {}
        assert s.label_for(1) is None


class TestZoneGating:
    """the structural fix: approach/exit frames (off-center) must not vote,
    so a track that is wrong off-zone and right in-zone resolves to the right
    label at the centerline — the diagnosis from the baseline regression."""

    def test_off_zone_frames_do_not_vote_or_delay_commit(self):
        s = TrackLabelStabilizer()  # zone_frac=0.15 -> zone_half=30 around mid=100
        # many OFF-zone good@0.8 frames: cx=10 -> |10-100|=90 > 30 -> ignored
        for _ in range(50):
            s.update([_det(1, "good", 0.8, cx=10)], FRAME_WIDTH)
        # off-zone: no votes at all
        assert s.label_for(1) is None
        assert s.frames_seen == {}
        assert s.votes == {}

        # now inject in-zone no_cap@0.9 frames (cx=100, in-zone)
        for _ in range(4):
            s.update([_det(1, "no_cap", 0.9, cx=100)], FRAME_WIDTH)
        # committed to no_cap; off-zone good frames never entered the vote
        assert s.committed[1] == "no_cap"
        assert s.label_for(1) == "no_cap"
        assert "good" not in s.votes[1]

    def test_off_zone_only_track_produces_no_votes(self):
        s = TrackLabelStabilizer()
        for _ in range(100):
            s.update([_det(1, "good", 0.95, cx=5)], FRAME_WIDTH)
        # wholly off-zone -> label_for None, no frames_seen, no committed
        assert s.label_for(1) is None
        assert s.frames_seen == {}
        assert s.votes == {}
        assert s.committed == {}

    def test_boundary_in_zone_votes_just_inside(self):
        # cx at exactly zone_half from center should vote (<=)
        s = TrackLabelStabilizer()  # zone_half = 0.15*200 = 30; cx=70 or 130
        s.update([_det(1, "no_cap", 0.99, cx=70)], FRAME_WIDTH)  # |70-100|=30
        s.update([_det(1, "no_cap", 0.99, cx=130)], FRAME_WIDTH)  # |130-100|=30
        assert s.frames_seen[1] == 2

    def test_boundary_just_outside_does_not_vote(self):
        s = TrackLabelStabilizer()  # zone_half = 30
        s.update([_det(1, "no_cap", 0.99, cx=69)], FRAME_WIDTH)  # 31 > 30
        assert s.frames_seen == {}


class TestInspectorIntegration:
    """mirror tests/test_inspector.py style: drive Inspector.process() with a
    synthetic flickering track crossing the centerline. the logged defect_type
    is the stabilized label while raw_defect_type preserves the instantaneous
    per-frame class."""

    def test_flickering_track_logs_stabilized_label(self):
        insp = Inspector()
        insp.start_session()
        frame_width = 200  # mid_x=100, tolerance=15, zone_half=30

        # phase 1: track enters OFF-evidence-zone (cx=10, |10-100|=90 > 30),
        # the model says "good" for many frames — these must NOT vote.
        for _ in range(6):
            insp.process(
                [
                    {
                        "bbox": (10, 0, 2, 10),  # cx=11, off-zone AND off-centerline
                        "confidence": 0.8,
                        "class_id": 0,
                        "defect_type": "good",
                        "track_id": 7,
                        "bottle_id": "BTL_00007",
                    }
                ],
                frame_width,
            )
        assert insp._labels.label_for(7) is None  # no votes from off-zone

        # phase 2: track moves INTO evidence zone, model says no_cap reliably.
        # cx=99 is on-centerline (|99-100|=1 <= 15) and in-zone (<= 30).
        # cx=81: in-zone (|81-100|=19 <= 30) but off-centerline (> 15)
        for _ in range(4):
            insp.process(
                [
                    {
                        "bbox": (80, 0, 2, 10),
                        "confidence": 0.9,
                        "class_id": 2,
                        "defect_type": "no_cap",
                        "track_id": 7,
                        "bottle_id": "BTL_00007",
                    }
                ],
                frame_width,
            )
        assert insp._labels.committed[7] == "no_cap"

        # nothing counted/logged yet (phase-2 frames were off-centerline)
        assert insp.total_inspected == 0
        assert insp.total_defects == 0

        # phase 3: track reaches the centerline; per-frame class flickers to
        # "good" for one frame (the documented near-miss pattern) — the
        # committed label wins and the crossing is logged as no_cap, while
        # raw_defect_type preserves the instantaneous blip (good).
        blip_dets = [
            {
                "bbox": (99, 0, 2, 10),  # cx=100, on centerline
                "confidence": 0.95,
                "class_id": 0,
                "defect_type": "good",  # instantaneous blip
                "track_id": 7,
                "bottle_id": "BTL_00007",
            }
        ]
        insp.process(blip_dets, frame_width)

        det = blip_dets[0]
        assert det["defect_type"] == "no_cap"
        assert det["raw_defect_type"] == "good"
        assert det.get("logged") is True
        assert insp.total_defects == 1
        assert insp.total_inspected == 1
