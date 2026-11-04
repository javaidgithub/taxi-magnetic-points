"""Stage 2 - recursive K-Means clustering and day-wise sub-clustering.

Ports cells 3-19 and 28 of
``notebooks/original/Magnetic_Points_Model_Updated_with_timeComplexity.ipynb``.

Algorithm
---------
1. K-Means (k = ``n_initial_clusters``, default 10) on ``[VehLat, VehLng]``.
2. The most populated cluster is re-clustered with a second K-Means
   (k = ``n_sub_clusters``, default 10). Its new labels are shifted by
   ``n_initial_clusters`` so they cannot collide with labels 0-9 and then
   replace the original label. The remaining 9 initial clusters are untouched,
   giving 9 + 10 = 19 spatial clusters (labels 1-9 and 10-19 in the paper's data,
   where the split cluster was label 0).
3. ``SubCluster`` = ``DayOfWeek``. This is a *label assignment*, not another
   clustering run: every (Cluster, day) pair is one of the paper's "133
   spatio-temporal segments" (19 x 7).
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

DEFAULT_FEATURES = ("VehLat", "VehLng")


def _kmeans(n_clusters: int, random_state: int, n_init=None) -> KMeans:
    """KMeans with the notebook's settings. ``n_init=None`` keeps scikit-learn's default
    (which changed from 10 to "auto" in scikit-learn 1.4 and can change labels)."""
    kwargs = {} if n_init is None else {"n_init": n_init}
    return KMeans(n_clusters=n_clusters, random_state=random_state, **kwargs)


def initial_kmeans(df: pd.DataFrame, n_clusters: int = 10, random_state: int = 42,
                   features: Sequence[str] = DEFAULT_FEATURES, n_init=None) -> np.ndarray:
    """Fit K-Means on the spatial features and return one label per row."""
    model = _kmeans(n_clusters, random_state, n_init)
    model.fit(df[list(features)])
    return model.labels_


def choose_split_cluster(labels: Union[np.ndarray, pd.Series]) -> int:
    """Return the label of the most populated cluster."""
    return int(pd.Series(labels).value_counts().idxmax())


def refine_cluster(df: pd.DataFrame, labels: np.ndarray, target: int, n_sub_clusters: int,
                   label_offset: int, random_state: int = 42,
                   features: Sequence[str] = DEFAULT_FEATURES, n_init=None) -> np.ndarray:
    """Re-cluster one cluster with K-Means and return updated labels.

    Rows whose label equals ``target`` receive ``sub_label + label_offset``;
    all other rows keep their label.
    """
    labels = np.array(labels, copy=True)
    mask = labels == target
    if mask.sum() < n_sub_clusters:
        raise ValueError(
            f"Cluster {target} has only {int(mask.sum())} points; cannot split into "
            f"{n_sub_clusters} sub-clusters."
        )
    model = _kmeans(n_sub_clusters, random_state, n_init)
    model.fit(df.loc[mask, list(features)])
    labels[mask] = model.labels_ + label_offset
    return labels


def assign_day_subcluster(df: pd.DataFrame) -> pd.Series:
    """Day-wise sub-cluster id = DayOfWeek (0=Monday ... 6=Sunday)."""
    return df["DayOfWeek"].astype(int)


def run_clustering(df: pd.DataFrame, n_initial_clusters: int = 10, n_sub_clusters: int = 10,
                   split_cluster: Union[str, int] = "auto", random_state: int = 42,
                   features: Sequence[str] = DEFAULT_FEATURES, n_init=None) -> pd.DataFrame:
    """Full clustering stage. Returns a copy of ``df`` with ``Cluster`` and ``SubCluster``.

    Parameters
    ----------
    split_cluster : "auto" or int
        "auto" splits the most populated initial cluster; an int forces that label
        (the original notebook hard-coded 0, which was the largest cluster).
    n_init : int, optional
        Number of K-Means restarts. ``None`` = scikit-learn default (see ``_kmeans``).
    """
    out = df.copy().reset_index(drop=True)
    labels = initial_kmeans(out, n_initial_clusters, random_state, features, n_init)
    target = choose_split_cluster(labels) if split_cluster == "auto" else int(split_cluster)
    labels = refine_cluster(out, labels, target, n_sub_clusters,
                            label_offset=n_initial_clusters,
                            random_state=random_state, features=features, n_init=n_init)
    out["Cluster"] = labels.astype(int)
    out["SubCluster"] = assign_day_subcluster(out)
    return out


def cluster_sizes(df: pd.DataFrame) -> pd.Series:
    """Points per final cluster, sorted by cluster label."""
    return df["Cluster"].value_counts().sort_index()


def segment_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Points per (Cluster, SubCluster/day) segment, with empty segments listed as 0."""
    counts = df.groupby(["Cluster", "SubCluster"]).size().unstack(fill_value=0)
    return counts.reindex(columns=range(7), fill_value=0)
