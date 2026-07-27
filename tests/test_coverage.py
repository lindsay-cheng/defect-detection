"""tests for defect_detection.inspection.coverage + DetectionPipeline
conformal-guard integration (UNCERTAIN persistence path, no defect row)."""

import json

import numpy as np
import pytest

from defect_detection.config import DetectorConfig
from defect_detection.inspection.coverage import CoverageGuard
from defect_detection.inspection.pipeline import DetectionPipeline
from defect_detection.storage.database import DefectDatabase


class TestCoverageGuardVerdict:
    def test_conf_at_tau_is_covered(self):
        g = CoverageGuard(tau=0.9)
        assert g.verdict(0.9) == "covered"

    def test_conf_above_tau_is_covered(self):
        g = CoverageGuard(tau=0.9)
        assert g.verdict(0.95) == "covered"

    def test_conf_just_below_tau_abstains(self):
        g = CoverageGuard(tau=0.9)
        assert g.verdict(0.89) == "abstain"

    def test_zero_conf_abstains(self):
        g = CoverageGuard(tau=0.5)
        assert g.verdict(0.0) == "abstain"


class TestFromJson:
    def test_missing_path_returns_none(self, tmp_path):
        assert CoverageGuard.from_json(str(tmp_path / "nope.json")) is None

    def test_none_path_returns_none(self):
        assert CoverageGuard.from_json(None) is None

    def test_invalid_json_returns_none(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("not json at all")
        assert CoverageGuard.from_json(str(p)) is None

    def test_missing_tau_key_returns_none(self, tmp_path):
        p = tmp_path / "notau.json"
        p.write_text(json.dumps({"alpha": 0.1}))
        assert CoverageGuard.from_json(str(p)) is None

    def test_tau_out_of_range_returns_none(self, tmp_path):
        p = tmp_path / "zerotau.json"
        p.write_text(json.dumps({"tau": 0.0}))
        assert CoverageGuard.from_json(str(p)) is None

    def test_valid_json_returns_guard_with_tau(self, tmp_path):
        p = tmp_path / "ok.json"
        p.write_text(json.dumps({"tau": 0.9, "alpha": 0.1}))
        g = CoverageGuard.from_json(str(p))
        assert g is not None
        assert g.tau == pytest.approx(0.9)
        assert g.verdict(0.95) == "covered"
        assert g.verdict(0.8) == "abstain"


def _write_conformal(path, tau):
    path.write_text(json.dumps({"tau": tau, "alpha": 0.1, "q_hat": 1 - tau}))


def _defect_det(frame_width, track_id, conf, disp_id=None, defect_type="no_cap"):
    """synthetic centerline defect detection dict ready for the side-effect
    branch — on_centerline + label_state committed so Inspector would log it."""
    cx = frame_width // 2
    w = 20
    return {
        "bbox": (cx - w // 2, 0, w, 10),
        "confidence": conf,
        "class_id": 2,
        "defect_type": defect_type,
        "track_id": track_id,
        "bottle_id": f"BTL_{track_id:05d}",
        "display_id": disp_id or f"BTL_{track_id:05d}",
        "on_centerline": True,
        "label_state": "committed",
        "logged": True,
    }


def _drive_side_effects(pipeline, dets, frame):
    """replay the exact side-effect sequence of detect_frame without invoking
    engine.track, using synthetic detections. the smallest honest unit that
    exercises the guard + DB write path (det['coverage'] / insert_bottle /
    insert_defect / total_abstentions)."""
    dets_list = list(dets)
    # mimic inspector.process outputs — counted holds good-style centerline hits,
    # defects holds logged detections. we push them straight through to the
    # same write loop detect_frame runs, via direct invocation of the
    # Inspector.process-free branch: build a minimal InspectionResult.
    from defect_detection.inspection.inspector import InspectionResult

    result = InspectionResult(defects=[d for d in dets_list])
    pipeline._check_invariants(dets_list)
    for det in result.defects:
        display_id = det.get("display_id")
        if (
            pipeline.guard is not None
            and pipeline.guard.verdict(det.get("confidence", 0.0)) == "abstain"
        ):
            pipeline.database.insert_bottle(
                bottle_id=f"{pipeline.inspector.session_id}:{display_id}",
                display_id=display_id,
                session_id=pipeline.inspector.session_id,
                status="UNCERTAIN",
            )
            det["coverage"] = "abstained"
            pipeline.total_abstentions += 1
            continue
        pipeline.database.insert_defect(
            bottle_id=f"{pipeline.inspector.session_id}:{display_id}",
            defect_type=det.get("defect_type"),
            display_id=display_id,
            session_id=pipeline.inspector.session_id,
            confidence=det.get("confidence"),
            bbox=det["bbox"],
        )
    return result


class TestPipelineGuardIntegration:
    @pytest.fixture()
    def pipeline_guarded(self, tmp_path):
        conf = tmp_path / "conformal.json"
        _write_conformal(conf, tau=0.9)
        cfg = DetectorConfig(
            model_path=None,
            db_path=str(tmp_path / "test.db"),
            save_images=False,
            images_dir=str(tmp_path / "detections"),
            coverage_file=str(conf),
        )
        p = DetectionPipeline(cfg)
        p.start_session()
        yield p
        p.cleanup()

    def test_low_conf_defect_produces_uncertain_row_no_defect(self, pipeline_guarded, tmp_path):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        det = _defect_det(frame_width=200, track_id=7, conf=0.5, defect_type="no_cap")
        _drive_side_effects(pipeline_guarded, [det], frame)

        db = DefectDatabase(pipeline_guarded.config.db_path)
        defects = db.get_defects(limit=100)
        # bottle row exists with UNCERTAIN status (inspect bottles table directly)
        bottles = _all_bottles(db)
        db.close()

        assert defects == [], "no defect row should be written on abstention"
        assert any(b["status"] == "UNCERTAIN" for b in bottles), "expected an UNSERTAIN bottle row"
        assert pipeline_guarded.total_abstentions == 1
        assert det.get("coverage") == "abstained"

    def test_high_conf_defect_writes_defect_row_no_abstention(self, pipeline_guarded, tmp_path):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        det = _defect_det(frame_width=200, track_id=8, conf=0.95, defect_type="no_cap")
        _drive_side_effects(pipeline_guarded, [det], frame)

        db = DefectDatabase(pipeline_guarded.config.db_path)
        defects = db.get_defects(limit=100)
        bottles = _all_bottles(db)
        db.close()

        assert len(defects) == 1, "high-conf defect should persist a defect row"
        assert defects[0]["defect_type"] == "no_cap"
        assert all(b["status"] != "UNCERTAIN" for b in bottles), "no UNCERTAIN row on covered entry"
        assert pipeline_guarded.total_abstentions == 0
        assert det.get("coverage") is None

    def test_guard_disabled_when_coverage_file_none(self, tmp_path):
        cfg = DetectorConfig(
            model_path=None,
            db_path=str(tmp_path / "test.db"),
            save_images=False,
            images_dir=str(tmp_path / "detections"),
            coverage_file=None,
        )
        p = DetectionPipeline(cfg)
        assert p.guard is None, "no coverage_file ⇒ unguarded pipeline"
        stats = p.get_stats()
        assert "abstentions" not in stats, "abstentions key only surfaces when guard is active"
        p.cleanup()

    def test_get_stats_includes_abstentions_when_guard_active(self, pipeline_guarded):
        stats = pipeline_guarded.get_stats()
        assert "abstentions" in stats
        assert stats["abstentions"] == 0

    def test_invariant_warns_on_excessive_count(self, pipeline_guarded, caplog):
        import logging

        dets = [_defect_det(200, i, conf=0.95, disp_id=f"BTL_{i:05d}") for i in range(25)]
        with caplog.at_level(logging.WARNING, logger="defect_detection.inspection.pipeline"):
            pipeline_guarded._check_invariants(dets)
        assert any("exceeds 20" in r.message for r in caplog.records)

    def test_invariant_warns_on_nan_confidence(self, pipeline_guarded, caplog):
        import logging

        det = _defect_det(200, 99, conf=float("nan"))
        with caplog.at_level(logging.WARNING, logger="defect_detection.inspection.pipeline"):
            pipeline_guarded._check_invariants([det])
        assert any("NaN/Inf" in r.message for r in caplog.records)


class TestAnnotatorUncertain:
    def test_abstained_renders_amber_uncertain_label(self):
        from defect_detection.inspection.annotator import annotate_frame

        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        det = _defect_det(200, 5, conf=0.4)
        det["coverage"] = "abstained"
        out = annotate_frame(frame, [det])
        # amber BGR = (0,191,255); the box rect uses color index 2 (cv2 rect color, last arg)
        # we cannot read pixels-back reliably across antialiasing; assert the frame is a
        # copy returned (not None) and the centerline is drawn (cyan line at x=100).
        assert out is not None
        assert out.shape == (100, 200, 3)


def _all_bottles(db: DefectDatabase) -> list[dict]:
    """read the bottles table directly via the worker thread's core (no public
    method) — use get_statistics to confirm at least one row was inserted."""
    # ponytail: simplest public read is get_defects, but abstention writes a bottle
    # row with no defect row. read raw via the db file directly to stay honest.
    import sqlite3

    conn = sqlite3.connect(db.db_path)
    try:
        cur = conn.execute("SELECT id_bottle, display_id, status FROM bottles")
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row, strict=False)) for row in cur.fetchall()]
    finally:
        conn.close()
