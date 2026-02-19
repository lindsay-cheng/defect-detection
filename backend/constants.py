"""
shared constants for the defect-detection backend
"""

DEFAULT_DB_PATH = "database/defects.db"
DEFAULT_CONF_THRESHOLD = 0.5

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"

DEFECT_TYPE_GOOD = "good"


def get_display_id(detection: dict, default: str = "N/A") -> str:
    """resolve the operator-facing id from a detection dict, with fallback"""
    return detection.get("display_id") or detection.get("bottle_id", default)


def make_db_key(session_id: str, display_id: str | None, track_id=None) -> str:
    """build the composite key used as bottles.id_bottle"""
    if display_id:
        return f"{session_id}:{display_id}"
    return f"{session_id}:TRK_{track_id}"
