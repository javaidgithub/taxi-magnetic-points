"""Stage 3 - cluster-quality metrics and supervised-classifier evaluation.

Ports notebook cells 21-27 (KNN / Decision Tree / Random Forest, stratified
10-fold CV, weighted precision/recall/F1, confusion matrices).

What the classifiers measure
----------------------------
The cluster labels were produced by K-Means on the same two features
(``VehLat``, ``VehLng``) that the classifiers receive. High accuracy therefore
shows that the K-Means partition is *learnable* from coordinates; it is not an
independent measure of demand-prediction quality. See docs/DISCREPANCIES.md.
"""
from __future__ import annotations

import warnings
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, calinski_harabasz_score,
                             confusion_matrix, davies_bouldin_score,
                             f1_score, precision_score, recall_score,
                             silhouette_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def build_classifiers(knn_neighbors: int = 3, random_forest_trees: int = 100,
                      random_state: int = 42, n_jobs: int = 1) -> Dict[str, object]:
    """Classifiers with the hyper-parameters used in the notebook."""
    return {
        "K-NN": KNeighborsClassifier(n_neighbors=knn_neighbors),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=random_forest_trees, random_state=random_state, n_jobs=n_jobs),
    }


def cross_validate_classifier(model, X: pd.DataFrame, y: pd.Series, n_splits: int = 10,
                              random_state: int = 42):
    """Stratified K-fold CV; returns (mean metrics dict, out-of-fold predictions).

    Metric definitions match the notebook: accuracy plus *weighted* precision,
    recall and F1, each averaged over the folds. Predictions from every fold are
    pooled into one vector used for the confusion matrix (as
    ``cross_val_predict`` did in the notebook).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    X_arr = X.to_numpy()
    y_arr = y.to_numpy()
    oof = np.empty_like(y_arr)
    folds = {"Accuracy": [], "Precision": [], "Recall": [], "F1": []}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # tiny classes -> undefined precision in a fold
        for train_idx, test_idx in skf.split(X_arr, y_arr):
            model.fit(X_arr[train_idx], y_arr[train_idx])
            pred = model.predict(X_arr[test_idx])
            oof[test_idx] = pred
            folds["Accuracy"].append(accuracy_score(y_arr[test_idx], pred))
            folds["Precision"].append(precision_score(y_arr[test_idx], pred, average="weighted", zero_division=0))
            folds["Recall"].append(recall_score(y_arr[test_idx], pred, average="weighted", zero_division=0))
            folds["F1"].append(f1_score(y_arr[test_idx], pred, average="weighted", zero_division=0))
    return {k: float(np.mean(v)) for k, v in folds.items()}, oof


def evaluate_classifiers(df: pd.DataFrame, features: Sequence[str] = ("VehLat", "VehLng"),
                         label: str = "Cluster", n_splits: int = 10, random_state: int = 42,
                         knn_neighbors: int = 3, random_forest_trees: int = 100,
                         n_jobs: int = 1, models: Optional[Iterable[str]] = None):
    """Evaluate the three classifiers.

    Returns
    -------
    results : pandas.DataFrame
        One row per classifier (Accuracy, Precision, Recall, F1).
    confusion : dict[str, numpy.ndarray]
        Pooled out-of-fold confusion matrix per classifier.
    labels : list[int]
        Sorted class labels (axis order of the confusion matrices).
    """
    X, y = df[list(features)], df[label]
    labels = sorted(int(v) for v in y.unique())
    clfs = build_classifiers(knn_neighbors, random_forest_trees, random_state, n_jobs)
    if models is not None:
        clfs = {k: v for k, v in clfs.items() if k in set(models)}
    rows, confusion = [], {}
    for name, model in clfs.items():
        metrics, oof = cross_validate_classifier(model, X, y, n_splits, random_state)
        rows.append({"Classifier": name, **metrics})
        confusion[name] = confusion_matrix(y.to_numpy(), oof, labels=labels)
    return pd.DataFrame(rows).set_index("Classifier"), confusion, labels


def clustering_metrics(df: pd.DataFrame, features: Sequence[str] = ("VehLat", "VehLng"),
                       label: str = "Cluster", silhouette_sample_size: Optional[int] = 20000,
                       random_state: int = 42) -> Dict[str, float]:
    """Silhouette, Davies-Bouldin and Calinski-Harabasz scores of the final clusters.

    NEW code - the uploaded notebooks do not compute these values although the
    paper reports them (Silhouette 0.736, Davies-Bouldin 0.320,
    Calinski-Harabasz 2,556,892). Silhouette is O(n^2) in memory/time, so it is
    computed on a random sample when ``silhouette_sample_size`` is smaller than
    the data; Davies-Bouldin and Calinski-Harabasz use all points.
    Which features/scaling/labels the authors used is not documented, so values
    from this function are not guaranteed to equal the published ones.
    """
    X = df[list(features)].to_numpy()
    y = df[label].to_numpy()
    n = len(df)
    sample = silhouette_sample_size if (silhouette_sample_size and silhouette_sample_size < n) else None
    return {
        "silhouette": float(silhouette_score(X, y, sample_size=sample, random_state=random_state)),
        "silhouette_sample_size": int(sample) if sample else int(n),
        "davies_bouldin": float(davies_bouldin_score(X, y)),
        "calinski_harabasz": float(calinski_harabasz_score(X, y)),
        "n_points": int(n),
        "n_clusters": int(len(np.unique(y))),
    }
