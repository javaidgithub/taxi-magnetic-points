import numpy as np

from magnetic_points import clustering as cl


def test_nineteen_clusters(clustered):
    # 10 initial clusters - 1 split + 10 sub-clusters = 19
    assert clustered["Cluster"].nunique() == 19


def test_split_labels_are_offset_and_target_removed(clustered, preprocessed):
    labels0 = cl.initial_kmeans(preprocessed)
    target = cl.choose_split_cluster(labels0)
    final = set(clustered["Cluster"].unique())
    assert target not in final                       # the split cluster no longer exists
    assert set(range(10, 20)).issubset(final)        # sub-clusters carry labels 10..19
    untouched = set(range(10)) - {target}
    assert untouched.issubset(final)


def test_untouched_clusters_keep_their_members(clustered, preprocessed):
    labels0 = cl.initial_kmeans(preprocessed)
    target = cl.choose_split_cluster(labels0)
    keep = labels0 != target
    assert np.array_equal(clustered.loc[keep, "Cluster"].to_numpy(), labels0[keep])


def test_subcluster_is_day_of_week(clustered):
    assert (clustered["SubCluster"] == clustered["DayOfWeek"]).all()


def test_deterministic(preprocessed):
    a = cl.run_clustering(preprocessed)["Cluster"].to_numpy()
    b = cl.run_clustering(preprocessed)["Cluster"].to_numpy()
    assert np.array_equal(a, b)


def test_forced_split_cluster(preprocessed):
    out = cl.run_clustering(preprocessed, split_cluster=0)
    assert 0 not in set(out["Cluster"])


def test_too_small_cluster_raises(preprocessed):
    import pytest
    with pytest.raises(ValueError):
        cl.refine_cluster(preprocessed, np.zeros(len(preprocessed), dtype=int), target=5,
                          n_sub_clusters=10, label_offset=10)
