# Changelog

All notable changes are documented here. Format: [Keep a Changelog](https://keepachangelog.com).

## [0.1.0] - 2026-11-04

First public structuring of the research project.

### Added
- Package `magnetic_points` (preprocessing, clustering, evaluation, prediction, visualization, synthetic data, config, CLI).
- `MagneticPointPredictor` (vectorised port of the notebook's final predictor; output tested equal to the original).
- `evaluation.clustering_metrics` and `cluster-metrics` command (**new**; the paper's Silhouette / Davies-Bouldin / Calinski-Harabasz values had no code in the provided notebooks).
- Synthetic trip-log generator and sample file; 30 tests; docs, diagrams, figures; open-source files.
- `clustering.n_init` option to pin K-Means restarts across scikit-learn versions.

### Changed (relative to the original notebooks; originals kept in `notebooks/original/`)
- `Minutes` uses `dt.minute` (notebook used `dt.hour`); `TourId` is dropped.
- Pickups are kept with `Status == 6` instead of dropping `Status == 8`; warning on other codes.
- The cluster to split is chosen automatically (largest) instead of hard-coded `0`.
- Scatter plots use longitude on the x-axis.
- `SubCluster` is an integer column; map popup shows the actual time window.

### Not ported
- Notebook cell 35 (older predictor), cell 13 (hold-out KNN on cluster 0), cell 37 (ad-hoc filter).

See `docs/DISCREPANCIES.md` for the full list of paper/code differences.
