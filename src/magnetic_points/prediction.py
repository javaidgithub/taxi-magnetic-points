"""Stage 4 - "magnetic point" recommendation.

Ports notebook cell 36 (the final, faster version of the predictor).

How a recommendation is produced
--------------------------------
1. Keep only records of the requested ``DayOfWeek``.
2. For every (Cluster, SubCluster) group of that day compute the *centroid*
   (mean latitude / longitude of all its pickups, at any time of day).
3. Compute the haversine distance (km) from the driver to each centroid and
   return the ``num_neighbors`` nearest.
4. For each returned point count the pickups of that group whose time of day is
   within +/- ``time_window`` minutes of the requested time.

Note: the requested time influences only step 4 (the pickup count), not which
points are chosen. No trained classifier (KNN or other) is used here, although
the paper text states a KNN model was deployed - see docs/DISCREPANCIES.md #6.

The time window does not wrap around midnight (23:55 +/- 10 does not include
00:03), exactly as in the notebook.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088  # same constant as the `haversine` PyPI package


def haversine_km(lat1, lng1, lat2, lng2) -> np.ndarray:
    """Great-circle distance in km. Accepts scalars or numpy arrays (vectorised)."""
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    a = (np.sin((lat2 - lat1) / 2.0) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2.0) ** 2)
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


class MagneticPointPredictor:
    """Recommend the nearest pickup hotspots ("magnetic points") for a driver.

    Parameters
    ----------
    df : pandas.DataFrame
        Clustered table with ``VehLat, VehLng, Hour, Minutes, DayOfWeek,
        Cluster, SubCluster`` (output of :func:`magnetic_points.clustering.run_clustering`).

    Examples
    --------
    >>> predictor = MagneticPointPredictor(clustered_df)          # doctest: +SKIP
    >>> predictor.predict(59.909851, 10.743474, hour=14, minute=30, day_of_week=1)  # doctest: +SKIP
    """

    REQUIRED = ["VehLat", "VehLng", "Hour", "Minutes", "DayOfWeek", "Cluster", "SubCluster"]

    def __init__(self, df: pd.DataFrame):
        missing = [c for c in self.REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"Clustered table is missing columns: {missing}")
        data = df.copy()
        data["_minute_of_day"] = data["Hour"].astype(int) * 60 + data["Minutes"].astype(int)
        self._centroids = (
            data.groupby(["DayOfWeek", "Cluster", "SubCluster"])[["VehLat", "VehLng"]]
            .mean().reset_index()
        )
        # sorted minute-of-day arrays make window counting O(log n) per group
        self._minutes = {
            key: np.sort(g["_minute_of_day"].to_numpy())
            for key, g in data.groupby(["DayOfWeek", "Cluster", "SubCluster"])
        }

    def predict(self, latitude: float, longitude: float, hour: int, minute: int,
                day_of_week: int, num_neighbors: int = 3,
                time_window: int = 10) -> pd.DataFrame:
        """Return the ``num_neighbors`` nearest hotspots.

        Returns
        -------
        pandas.DataFrame
            Columns ``Cluster, SubCluster, VehLat, VehLng, Distance_km,
            Pickups_in_window`` ordered by distance. ``result.attrs['elapsed_seconds']``
            holds the wall-clock time of the call.
        """
        start = time.perf_counter()
        self._validate(latitude, longitude, hour, minute, day_of_week)
        day = self._centroids[self._centroids["DayOfWeek"] == day_of_week]
        if day.empty:
            raise ValueError(f"No pickups recorded for DayOfWeek={day_of_week}.")
        dist = haversine_km(latitude, longitude, day["VehLat"].to_numpy(), day["VehLng"].to_numpy())
        order = np.argsort(dist, kind="stable")[:num_neighbors]
        chosen = day.iloc[order].copy()
        chosen["Distance_km"] = dist[order]
        t = hour * 60 + minute
        counts = []
        for _, row in chosen.iterrows():
            arr = self._minutes[(day_of_week, row["Cluster"], row["SubCluster"])]
            counts.append(int(np.searchsorted(arr, t + time_window, side="right")
                              - np.searchsorted(arr, t - time_window, side="left")))
        chosen["Pickups_in_window"] = counts
        chosen = chosen.drop(columns="DayOfWeek").reset_index(drop=True)
        chosen["Cluster"] = chosen["Cluster"].astype(int)
        chosen["SubCluster"] = chosen["SubCluster"].astype(int)
        chosen.attrs["elapsed_seconds"] = time.perf_counter() - start
        return chosen

    @staticmethod
    def _validate(lat, lng, hour, minute, day):
        if not -90 <= lat <= 90 or not -180 <= lng <= 180:
            raise ValueError("latitude must be in [-90, 90] and longitude in [-180, 180]")
        if not 0 <= hour <= 23 or not 0 <= minute <= 59:
            raise ValueError("hour must be 0-23 and minute 0-59")
        if day not in range(7):
            raise ValueError("day_of_week must be 0 (Monday) .. 6 (Sunday)")
