"""Configuration loading.

Reads ``configs/default.yaml`` (or a user supplied YAML file) and deep-merges it
over built-in defaults, so a custom file only needs the keys it changes.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

DEFAULTS: Dict[str, Any] = {
    "paths": {
        "raw_csv": "data/raw/TripLog.csv",
        "preprocessed_csv": "data/processed/TripLog_DataSet.csv",
        "clustered_csv": "data/processed/TripLog_Clustered.csv",
        "output_dir": "outputs",
    },
    "preprocessing": {"pickup_status": 6, "datetime_column": "CreatedDate"},
    "clustering": {
        "n_initial_clusters": 10,
        "n_sub_clusters": 10,
        "split_cluster": "auto",
        "random_state": 42,
        "n_init": None,
        "features": ["VehLat", "VehLng"],
    },
    "evaluation": {
        "n_splits": 10,
        "random_state": 42,
        "features": ["VehLat", "VehLng"],
        "label": "Cluster",
        "knn_neighbors": 3,
        "random_forest_trees": 100,
        "n_jobs": 1,
    },
    "cluster_metrics": {"silhouette_sample_size": 20000, "random_state": 42},
    "prediction": {"num_neighbors": 3, "time_window_minutes": 10},
    "visualization": {"map_zoom_recommendation": 13, "map_zoom_overview": 12},
}


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Return the merged configuration dictionary.

    Parameters
    ----------
    path : str, optional
        YAML file. If omitted, built-in defaults are returned.
    """
    if path is None:
        return copy.deepcopy(DEFAULTS)
    with open(Path(path), "r", encoding="utf-8") as fh:
        user = yaml.safe_load(fh) or {}
    return _merge(DEFAULTS, user)
