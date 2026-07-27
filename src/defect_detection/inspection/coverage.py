"""conformal coverage guard: threshold-only abstention at the log decision.

loads the calibrated threshold (tau) from benchmarks/results/conformal.json
and decides whether a detection's top-class confidence is covered by the
split-conformal guarantee. used by DetectionPipeline so a low-conf defect
at the crossing is logged as UNCERTAIN (no defect row) instead of being
persisted as a raw-threshold cut. # ponytail: threshold-only (top-class
softmax); upgrade = raw-logit RAPS if ultralytics ever exposes per-class
vectors post-NMS (see vault alternatives ledger).
"""

from __future__ import annotations

import json
import logging
import os

log = logging.getLogger(__name__)


class CoverageGuard:
    """threshold-only conformal abstention guard."""

    def __init__(self, tau: float):
        self.tau = float(tau)

    def verdict(self, conf: float) -> str:
        """'covered' if conf >= tau (log the defect), else 'abstain'."""
        return "covered" if conf >= self.tau else "abstain"

    @classmethod
    def from_json(cls, path: str) -> CoverageGuard | None:
        """load tau from a conformal.json; None on missing/invalid file
        (graceful disable — the pipeline runs unguarded rather than failing)."""
        if not path or not os.path.exists(path):
            log.info("coverage guard disabled — no conformal.json at %s", path)
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            tau = float(data["tau"])
        except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            log.warning("coverage guard disabled — invalid %s: %s", path, e)
            return None
        if not (0.0 < tau < 1.0):
            log.warning("coverage guard disabled — tau=%r out of (0,1) range; refusing", tau)
            return None
        return cls(tau)
