import pytest

from magnetic_points import clustering, preprocessing
from magnetic_points.synthetic import generate_trip_log


@pytest.fixture(scope="session")
def raw():
    return generate_trip_log(n_trips=3000, seed=7)


@pytest.fixture(scope="session")
def preprocessed(raw):
    return preprocessing.preprocess(raw)


@pytest.fixture(scope="session")
def clustered(preprocessed):
    return clustering.run_clustering(preprocessed)
