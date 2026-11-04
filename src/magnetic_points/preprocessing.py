"""Stage 1 - data preprocessing and datetime feature engineering.

Ports ``notebooks/original/Magnetic_Points_PreProcessing.ipynb``.

Input  : raw log with columns ``TourId, VehLat, VehLng, Status, CreatedDate``.
Output : pickup-only table with columns
         ``VehLat, VehLng, Hour, Minutes, DayOfWeek``.

Differences from the uploaded notebook (all logged in docs/DISCREPANCIES.md):

* ``Minutes`` is taken from ``dt.minute``. The uploaded notebook used
  ``dt.hour`` for this column (a bug); the dataset used by the model notebook
  contains real minute values, so this matches the data actually modelled.
* ``TourId`` is dropped, as the paper describes (the notebook kept it).
* Pickups are selected with ``Status == pickup_status`` (the paper's wording)
  instead of dropping ``Status == 8``. A warning is emitted if the data contains
  Status codes other than 6/8, where the two approaches would differ.
"""
from __future__ import annotations

import warnings
from typing import Dict

import pandas as pd

RAW_COLUMNS = ["TourId", "VehLat", "VehLng", "Status", "CreatedDate"]
PREPROCESSED_COLUMNS = ["VehLat", "VehLng", "Hour", "Minutes", "DayOfWeek"]


def load_raw(path: str) -> pd.DataFrame:
    """Read the raw trip log CSV and verify the expected columns exist."""
    df = pd.read_csv(path)
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Raw file {path!r} is missing required columns {missing}. "
            f"Expected columns: {RAW_COLUMNS}. Found: {list(df.columns)}"
        )
    return df


def filter_pickups(df: pd.DataFrame, pickup_status: int = 6) -> pd.DataFrame:
    """Keep only pickup records (``Status == pickup_status``)."""
    statuses = set(df["Status"].unique())
    unexpected = statuses - {pickup_status, 8}
    if unexpected:
        warnings.warn(
            f"Status values other than {pickup_status}/8 found: {sorted(unexpected)}. "
            "The original notebook dropped only Status==8, so these rows would "
            "have been kept there; this function drops them.",
            stacklevel=2,
        )
    return df[df["Status"] == pickup_status].copy()


def add_datetime_features(df: pd.DataFrame, datetime_column: str = "CreatedDate") -> pd.DataFrame:
    """Add ``Hour`` (0-23), ``Minutes`` (0-59), ``DayOfWeek`` (0=Monday..6=Sunday)."""
    ts = pd.to_datetime(df[datetime_column])
    out = df.copy()
    out["Hour"] = ts.dt.hour
    out["Minutes"] = ts.dt.minute
    out["DayOfWeek"] = ts.dt.day_of_week
    return out


def preprocess(df: pd.DataFrame, pickup_status: int = 6,
               datetime_column: str = "CreatedDate") -> pd.DataFrame:
    """Run the complete preprocessing stage on an in-memory raw table."""
    out = filter_pickups(df, pickup_status)
    out = add_datetime_features(out, datetime_column)
    out = out[PREPROCESSED_COLUMNS].reset_index(drop=True)
    return out


def summarize(raw: pd.DataFrame, processed: pd.DataFrame) -> Dict[str, int]:
    """Row counts before/after, for logging and for comparison with the paper."""
    return {"raw_rows": int(len(raw)), "pickup_rows": int(len(processed))}
