import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from magnetic_points import evaluation as ev


def test_matches_notebook_cross_val_score(clustered):
    """The single-pass loop must equal the notebook's four cross_val_score calls."""
    X, y = clustered[["VehLat", "VehLng"]], clustered["Cluster"]
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for name, model in [("K-NN", KNeighborsClassifier(n_neighbors=3)),
                        ("Decision Tree", DecisionTreeClassifier(random_state=42))]:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            expected = {
                "Accuracy": cross_val_score(model, X, y, cv=kfold, scoring="accuracy").mean(),
                "Precision": cross_val_score(model, X, y, cv=kfold, scoring="precision_weighted").mean(),
                "Recall": cross_val_score(model, X, y, cv=kfold, scoring="recall_weighted").mean(),
                "F1": cross_val_score(model, X, y, cv=kfold, scoring="f1_weighted").mean(),
            }
        got, _ = ev.cross_validate_classifier(model, X, y)
        for k in expected:
            assert np.isclose(got[k], expected[k]), (name, k)


def test_confusion_matrix_totals(clustered):
    res, cm, labels = ev.evaluate_classifiers(clustered, models=["K-NN"])
    assert cm["K-NN"].sum() == len(clustered)
    assert cm["K-NN"].shape == (len(labels), len(labels))
    assert 0.5 < res.loc["K-NN", "Accuracy"] <= 1.0


def test_clustering_metrics_keys_and_ranges(clustered):
    m = ev.clustering_metrics(clustered, silhouette_sample_size=1000)
    assert -1 <= m["silhouette"] <= 1
    assert m["davies_bouldin"] > 0 and m["calinski_harabasz"] > 0
    assert m["n_clusters"] == 19
