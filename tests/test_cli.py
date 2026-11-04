import json

from magnetic_points.cli import main


def test_full_pipeline_via_cli(tmp_path, raw):
    raw_csv = tmp_path / "raw.csv"
    raw.to_csv(raw_csv, index=False)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"paths:\n  preprocessed_csv: {tmp_path}/pre.csv\n  clustered_csv: {tmp_path}/clu.csv\n"
        f"  output_dir: {tmp_path}/out\nevaluation:\n  random_forest_trees: 10\n"
        "cluster_metrics:\n  silhouette_sample_size: 500\n")
    out = tmp_path / "out"
    assert main(["--config", str(cfg), "run-all", "--raw", str(raw_csv), "--out-dir", str(out)]) == 0
    for name in ["classifier_results.csv", "cluster_metrics.json", "clusters_map.html",
                 "nearest_subclusters_map.html", "knn_confusion_matrix.png", "clusters_scatter.png"]:
        assert (out / name).exists(), name
    assert json.loads((out / "cluster_metrics.json").read_text())["n_clusters"] == 19


def test_missing_input_returns_error(tmp_path):
    assert main(["preprocess", "--input", str(tmp_path / "nope.csv"), "--output", str(tmp_path / "o.csv")]) == 1
