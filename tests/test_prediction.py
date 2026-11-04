import numpy as np
import pytest
from haversine import haversine

from magnetic_points.prediction import MagneticPointPredictor, haversine_km

from reference_impl import notebook_predict

QUERIES = [
    (59.909851, 10.743474, 14, 30, 1, 10),
    (59.95, 10.80, 22, 5, 4, 10),
    (59.92, 10.70, 0, 3, 5, 30),
    (60.39, 5.33, 23, 59, 6, 10),     # far from the core
    (59.91, 10.75, 12, 0, 0, 0),      # zero-width window
]


def test_haversine_matches_package():
    for args in [(59.9, 10.7, 59.95, 10.8), (0, 0, 0, 1), (69.6, 18.9, 58.1, 7.9)]:
        assert haversine_km(*args) == pytest.approx(haversine(args[:2], args[2:]), rel=1e-12)


@pytest.mark.parametrize("lat,lng,hour,minute,day,window", QUERIES)
def test_matches_notebook_implementation(clustered, lat, lng, hour, minute, day, window):
    ref_nearest, ref_counts = notebook_predict(clustered, lat, lng, hour, minute, day, 3, window)
    got = MagneticPointPredictor(clustered).predict(lat, lng, hour, minute, day, 3, window)
    assert got["Cluster"].tolist() == ref_nearest["Cluster"].astype(int).tolist()
    assert np.allclose(got["Distance_km"], ref_nearest["Distance"].to_numpy())
    ref = [ref_counts[(c, s)] for c, s in zip(ref_nearest["Cluster"], ref_nearest["SubCluster"])]
    assert got["Pickups_in_window"].tolist() == ref


def test_results_sorted_by_distance(clustered):
    res = MagneticPointPredictor(clustered).predict(59.91, 10.75, 10, 0, 2)
    assert res["Distance_km"].is_monotonic_increasing
    assert len(res) == 3


@pytest.mark.parametrize("kwargs", [
    dict(latitude=95, longitude=10, hour=1, minute=1, day_of_week=1),
    dict(latitude=59, longitude=10, hour=24, minute=1, day_of_week=1),
    dict(latitude=59, longitude=10, hour=1, minute=60, day_of_week=1),
    dict(latitude=59, longitude=10, hour=1, minute=1, day_of_week=7),
])
def test_invalid_inputs_raise(clustered, kwargs):
    with pytest.raises(ValueError):
        MagneticPointPredictor(clustered).predict(**kwargs)


def test_missing_columns_raise(clustered):
    with pytest.raises(ValueError):
        MagneticPointPredictor(clustered.drop(columns="SubCluster"))
