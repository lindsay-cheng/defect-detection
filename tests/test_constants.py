"""tests for backend.constants helpers"""
from backend.constants import get_display_id, make_db_key


class TestGetDisplayId:
    def test_prefers_display_id(self):
        det = {"display_id": "BTL_00001", "bottle_id": "BTL_00099"}
        assert get_display_id(det) == "BTL_00001"

    def test_falls_back_to_bottle_id(self):
        det = {"bottle_id": "BTL_00099"}
        assert get_display_id(det) == "BTL_00099"

    def test_returns_default_when_both_missing(self):
        assert get_display_id({}) == "N/A"

    def test_custom_default(self):
        assert get_display_id({}, default="?") == "?"

    def test_none_display_id_falls_through(self):
        det = {"display_id": None, "bottle_id": "BTL_00005"}
        assert get_display_id(det) == "BTL_00005"


class TestMakeDbKey:
    def test_with_display_id(self):
        assert make_db_key("20260101_120000", "BTL_00001") == "20260101_120000:BTL_00001"

    def test_without_display_id_uses_track(self):
        assert make_db_key("20260101_120000", None, track_id=42) == "20260101_120000:TRK_42"

    def test_empty_display_id_uses_track(self):
        assert make_db_key("s", "", track_id=7) == "s:TRK_7"
