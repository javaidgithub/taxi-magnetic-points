import pandas as pd
import pytest

from magnetic_points import preprocessing as pp


def _tiny():
    return pd.DataFrame({
        "TourId": [1, 1, 2, 2],
        "VehLat": [59.9, 59.8, 59.7, 59.6],
        "VehLng": [10.7, 10.6, 10.5, 10.4],
        "Status": [6, 8, 6, 8],
        "CreatedDate": ["2024-01-01 00:01", "2024-01-01 00:20",
                        "2024-01-03 14:37", "2024-01-03 15:00"],
    })


def test_keeps_only_pickups_and_expected_columns():
    out = pp.preprocess(_tiny())
    assert list(out.columns) == pp.PREPROCESSED_COLUMNS
    assert len(out) == 2
    assert out["VehLat"].tolist() == [59.9, 59.7]


def test_minutes_are_minutes_not_hours():
    out = pp.preprocess(_tiny())
    assert out["Hour"].tolist() == [0, 14]
    assert out["Minutes"].tolist() == [1, 37]  # the notebook bug would give [0, 14]


def test_day_of_week_monday_is_zero():
    out = pp.preprocess(_tiny())
    assert out["DayOfWeek"].tolist() == [0, 2]  # 2024-01-01 Monday, 2024-01-03 Wednesday


def test_unexpected_status_warns():
    df = _tiny()
    df.loc[0, "Status"] = 7
    with pytest.warns(UserWarning):
        pp.preprocess(df)


def test_missing_columns_raise(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        pp.load_raw(str(p))
