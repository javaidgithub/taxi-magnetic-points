"""Synthetic stand-in for the proprietary AbsoluIT trip log.

The real data cannot be published. This generator produces a table with the
*same schema* (``TourId, VehLat, VehLng, Status, CreatedDate``) so the whole
pipeline can be exercised end to end. It is **not** a model of Norwegian taxi
demand and results computed on it say nothing about the paper's findings.

Shape of the synthetic data (loosely imitating the paper's description of a
dense urban core plus sparse far-away points):

* a dense core made of several Gaussian blobs around central Oslo,
* sparser suburban blobs,
* a few small, distant groups (other Norwegian cities).

Each trip yields two rows: a pickup (Status 6) then a drop-off (Status 8).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# (lat, lng, sigma_degrees, weight)
_CORE = [(59.9139, 10.7522, 0.006, 0.30), (59.9230, 10.7380, 0.008, 0.18),
         (59.9060, 10.7650, 0.007, 0.12), (59.9330, 10.7800, 0.010, 0.07)]
_SUBURBS = [(59.9500, 10.8000, 0.020, 0.06), (59.8600, 10.8000, 0.025, 0.05),
            (59.9300, 10.6500, 0.020, 0.05), (59.8900, 10.6000, 0.030, 0.04),
            (60.1930, 11.0980, 0.030, 0.03)]
_FAR = [(60.3913, 5.3221, 0.02, 0.006), (63.4305, 10.3951, 0.02, 0.006),
        (69.6492, 18.9553, 0.02, 0.006), (58.9700, 5.7331, 0.02, 0.006),
        (58.1467, 7.9956, 0.02, 0.006)]


def generate_trip_log(n_trips: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Return a raw-schema trip log with ``2 * n_trips`` rows."""
    rng = np.random.default_rng(seed)
    spots = _CORE + _SUBURBS + _FAR
    weights = np.array([s[3] for s in spots], dtype=float)
    weights /= weights.sum()
    comp = rng.choice(len(spots), size=n_trips, p=weights)
    lat = np.array([spots[c][0] for c in comp]) + rng.normal(0, 1, n_trips) * np.array([spots[c][2] for c in comp])
    lng = np.array([spots[c][1] for c in comp]) + rng.normal(0, 1, n_trips) * np.array([spots[c][2] for c in comp]) * 2.0

    # timestamps in 2024: weekend-heavy, evening/night-heavy
    start = pd.Timestamp("2024-01-01")
    day = rng.choice(366, size=n_trips)
    dow = (start + pd.to_timedelta(day, unit="D")).day_of_week.to_numpy()
    keep = rng.random(n_trips) < np.where(dow >= 4, 1.0, 0.55)
    hour_p = np.array([5, 4, 3, 2, 1, 1, 2, 3, 4, 3, 3, 3, 3, 3, 3, 3, 4, 5, 5, 5, 6, 7, 7, 6], dtype=float)
    hour_p /= hour_p.sum()
    hour = rng.choice(24, size=n_trips, p=hour_p)
    minute = rng.integers(0, 60, size=n_trips)
    pickup_ts = start + pd.to_timedelta(day, unit="D") + pd.to_timedelta(hour * 60 + minute, unit="m")
    dur = rng.integers(5, 40, size=n_trips)
    drop_ts = pickup_ts + pd.to_timedelta(dur, unit="m")
    drop_lat = lat + rng.normal(0, 0.01, n_trips)
    drop_lng = lng + rng.normal(0, 0.02, n_trips)

    tour = 1910629 + np.arange(n_trips)
    pick = pd.DataFrame({"TourId": tour, "VehLat": lat, "VehLng": lng, "Status": 6, "CreatedDate": pickup_ts})
    drop = pd.DataFrame({"TourId": tour, "VehLat": drop_lat, "VehLng": drop_lng, "Status": 8, "CreatedDate": drop_ts})
    pick, drop = pick[keep], drop[keep]
    out = pd.concat([pick, drop]).sort_values(["CreatedDate", "TourId", "Status"], kind="stable").reset_index(drop=True)
    out["CreatedDate"] = out["CreatedDate"].dt.strftime("%Y-%m-%d %H:%M")
    out["VehLat"] = out["VehLat"].round(6)
    out["VehLng"] = out["VehLng"].round(6)
    return out
