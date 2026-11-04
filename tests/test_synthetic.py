from magnetic_points.synthetic import generate_trip_log


def test_schema_and_reproducibility():
    a, b = generate_trip_log(500, seed=1), generate_trip_log(500, seed=1)
    assert list(a.columns) == ["TourId", "VehLat", "VehLng", "Status", "CreatedDate"]
    assert a.equals(b)
    assert set(a["Status"].unique()) == {6, 8}
