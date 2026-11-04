# Code reference

Package: `src/magnetic_points/`. Execution order:
`preprocessing → clustering → evaluation → prediction → visualization`
(`cli.py` wires them together).

## Mapping to the original notebooks

| Notebook (cell) | Repository |
|---|---|
| PreProcessing 1-11 | `preprocessing.preprocess` |
| Model 3-8 (initial K-Means) | `clustering.initial_kmeans` |
| Model 10-16 (recursive split) | `clustering.refine_cluster`, `clustering.run_clustering` |
| Model 28-31 (SubCluster) | `clustering.assign_day_subcluster`, `clustering.segment_counts` |
| Model 9, 12, 18, 19 (plots) | `visualization.plot_cluster_scatter`, `plot_points_by_day` |
| Model 21-27 (KNN / DT / RF) | `evaluation.evaluate_classifiers` |
| Model 33 (overview map) | `visualization.create_overview_map` |
| Model 35 (older predictor) | *not ported* (superseded) |
| Model 36 (final predictor + map) | `prediction.MagneticPointPredictor`, `visualization.create_recommendation_map` |
| Model 13 (hold-out KNN on cluster 0, acc. 0.9973) | *not ported*; not reported in the paper |
| Model 37 (ad-hoc filter of cluster 17 / day 1) | *not ported*; exploratory |
| – | `evaluation.clustering_metrics` **new** (paper metrics had no code) |

## `preprocessing.py`
Purpose: raw log → pickup table. Depends on: pandas.

| Function | Input | Output |
|---|---|---|
| `load_raw(path)` | CSV path | DataFrame; raises `ValueError` if a required column is missing |
| `filter_pickups(df, pickup_status=6)` | raw DataFrame | rows with that status; warns on unexpected codes |
| `add_datetime_features(df, datetime_column)` | DataFrame | copy + `Hour`, `Minutes`, `DayOfWeek` |
| `preprocess(df, pickup_status, datetime_column)` | raw DataFrame | `VehLat, VehLng, Hour, Minutes, DayOfWeek` (index reset) |
| `summarize(raw, processed)` | two DataFrames | row-count dict |

## `clustering.py`
Purpose: recursive K-Means and day label. Depends on: pandas, numpy, scikit-learn.

| Function | Input | Output |
|---|---|---|
| `initial_kmeans(df, n_clusters, random_state, features, n_init)` | preprocessed DataFrame | label array |
| `choose_split_cluster(labels)` | labels | label of the most populated cluster |
| `refine_cluster(df, labels, target, n_sub_clusters, label_offset, ...)` | labels | labels with `target` replaced by `sub + offset` |
| `assign_day_subcluster(df)` | DataFrame | `DayOfWeek` as int Series |
| `run_clustering(df, ...)` | preprocessed DataFrame | copy + `Cluster`, `SubCluster` |
| `cluster_sizes(df)` / `segment_counts(df)` | clustered DataFrame | size Series / cluster × day count table |

## `evaluation.py`
Purpose: classifier CV and cluster-quality scores. Depends on: scikit-learn.

| Function | Input | Output |
|---|---|---|
| `build_classifiers(...)` | hyper-parameters | dict name → estimator |
| `cross_validate_classifier(model, X, y, n_splits, random_state)` | estimator, data | (mean-metric dict, out-of-fold predictions) |
| `evaluate_classifiers(df, ..., models=None)` | clustered DataFrame | (results DataFrame, {name: confusion matrix}, sorted labels) |
| `clustering_metrics(df, ..., silhouette_sample_size)` | clustered DataFrame | dict: silhouette, Davies–Bouldin, Calinski–Harabasz, sizes |

## `prediction.py`
Purpose: recommend magnetic points. Depends on: numpy, pandas.

* `haversine_km(lat1, lng1, lat2, lng2)` – vectorised great-circle distance (R = 6371.0088 km).
* `MagneticPointPredictor(df)` – `df` needs `VehLat, VehLng, Hour, Minutes, DayOfWeek, Cluster, SubCluster`.
  * `.predict(latitude, longitude, hour, minute, day_of_week, num_neighbors=3, time_window=10)` → DataFrame `Cluster, SubCluster, VehLat, VehLng, Distance_km, Pickups_in_window`; elapsed time in `result.attrs["elapsed_seconds"]`. Raises `ValueError` for out-of-range inputs or a day with no data.

## `visualization.py`
Purpose: figures and maps. Depends on: matplotlib, seaborn, folium.
`plot_cluster_scatter`, `plot_cluster_sizes`, `plot_points_by_day`, `plot_confusion_matrix`, `create_overview_map`, `create_recommendation_map`.

## `synthetic.py`
`generate_trip_log(n_trips, seed)` – synthetic raw-schema data for demos and tests. Not a demand model.

## `config.py`
`load_config(path=None)` – merges a YAML file over built-in defaults (`DEFAULTS`).

## `cli.py`
Sub-commands: `make-sample`, `preprocess`, `cluster`, `evaluate`, `cluster-metrics`, `plots`, `predict`, `run-all`. Returns exit code 1 with an `error:` message on missing files or invalid values.

## Tests (`tests/`)
`reference_impl.py` holds literal ports of the notebook predictor used as an oracle; the other files test preprocessing, clustering, evaluation (equal to `cross_val_score`), prediction (equal to the notebook), the CLI end to end, and the synthetic generator.
