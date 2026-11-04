# Reproducibility and troubleshooting

## What has and has not been verified

| Claim | Status |
|---|---|
| Unit/integration tests (`python -m pytest`) | **Executed**: 30 passed on Python 3.12.3 |
| Full pipeline (`run-all`) on the **synthetic** sample | **Executed** |
| Refactored predictor equals the notebook's cell-36 function | **Tested** on synthetic data (5 queries) |
| Refactored CV equals the notebook's `cross_val_score` | **Tested** on synthetic data |
| Pipeline on the real AbsoluIT data | **Not executed** (data unavailable) |
| Paper's numbers (clustering metrics, prediction example) | **Not reproduced**; classifier table and cluster sizes match the *saved notebook outputs* only |

## Tested environment

Python 3.12.3 · numpy 2.5.4 · scikit-learn 1.9.1 · matplotlib 3.11.2 · seaborn 0.13.2 ·
folium 0.20.0 · PyYAML 6.0.3 · haversine 2.9.0 · pytest 9.2.0 · pandas 3.0.6 (development
environment) **and** pandas 3.1.0 (a clean virtual environment created with
`pip install -r requirements-dev.txt && pip install -e .`; 30 tests passed and `run-all`
completed there as well).

`pyproject.toml` declares `python >= 3.10` (CI workflow targets 3.10-3.12 but has not run yet) and lower bounds for libraries; versions other
than the above are **untested**. The notebooks ran on Google Colab (Python 3.12, numpy
2.0.2, folium 0.20.0, haversine 2.9.0 are visible in the saved output; scikit-learn's
version is not recorded).

## Reproducing the paper's experiments (authorized users)

1. Obtain `TripLog.csv` (see `DATASET.md`) and place it at `data/raw/TripLog.csv`.
2. `python -m magnetic_points run-all` (or run the stages one by one, see README).
3. Compare with the values the **notebook** printed on the original data:

| Quantity | Notebook output |
|---|---|
| raw rows / pickup rows | 242,509 / 121,422 |
| initial K-Means sizes (label: n) | 0: 101,017 · 5: 8,865 · 7: 3,349 · 8: 2,909 · 3: 2,103 · 2: 1,261 · 6: 719 · 1: 514 · 9: 477 · 4: 208 |
| final 19 cluster sizes | 17: 36,696 · 11: 31,406 · 10: 10,054 · 5: 8,865 · 15: 7,891 · 18: 4,368 · 12: 3,863 · 14: 3,360 · 7: 3,349 · 8: 2,909 · 13: 2,607 · 3: 2,103 · 2: 1,261 · 6: 719 · 16: 684 · 1: 514 · 9: 477 · 4: 208 · 19: 88 |
| 10-fold CV accuracy (all four metrics equal at 4 d.p.) | KNN 0.9979 · Decision Tree 0.9975 · Random Forest 0.9983 |
| runtime of the cell-36 predictor on Colab | 0.4733 s |

4. Expect possible differences in **label numbering** (and therefore in which label is
   "17" etc.) if your scikit-learn version differs from the Colab one, because K-Means'
   default `n_init` changed in scikit-learn 1.4. Try `clustering.n_init: 10` and `1` in the
   config. Cluster *sizes* as a multiset are the thing to compare.
5. Run `python -m magnetic_points cluster-metrics`. Published values: Silhouette 0.736,
   Davies–Bouldin 0.320, Calinski–Harabasz 2,556,892. These had no code in the uploaded
   notebooks; if yours differ, the unknowns are feature set, scaling, and silhouette
   sampling (this repo samples 20,000 points by default; set
   `cluster_metrics.silhouette_sample_size: null` for the exact value, which needs ~n² memory/time).
6. The paper's example (0.26 / 3.04 / 6.45 km; 40 / 24 / 12 pickups) used a driver position
   that is not documented in the notebook output; the notebook example input is
   `lat=59.909851, lng=10.743474, 14:30, Tuesday (day=1), window=10`.

Random seeds: K-Means `random_state=42`, CV `random_state=42`, Decision Tree and Random Forest
`random_state=42`, synthetic generator `seed=42`.

## Expected outputs of `run-all` (synthetic sample)

`outputs/` will contain `classifier_results.csv`, `cluster_metrics.json`, `knn_`, `decision_tree_`,
`random_forest_confusion_matrix.png`, `clusters_scatter.png`, `cluster_sizes.png`,
`points_by_day.png`, `clusters_map.html`, `nearest_subclusters_map.html`.
On the synthetic sample (5,196 pickups) the run printed 19 clusters, KNN / DT / RF accuracy of
0.9736 / 0.9754 / 0.9821, and metrics silhouette 0.2116, Davies–Bouldin 0.6668,
Calinski–Harabasz 484,637. **These numbers describe synthetic data only.**

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `error: [Errno 2] No such file or directory: 'data/raw/TripLog.csv'` | Raw data is not in the repo. Put it there, pass `--input/--raw`, or use `make-sample`. |
| `Raw file ... is missing required columns` | Header must contain `TourId, VehLat, VehLng, Status, CreatedDate`. |
| `UserWarning: Status values other than 6/8 found` | Your file has other status codes; they are dropped. The original notebook would have kept them. |
| `ValueError: Cluster N has only K points; cannot split` | The chosen cluster is smaller than `n_sub_clusters`; reduce it or use `split_cluster: auto`. |
| `ValueError: No pickups recorded for DayOfWeek=…` | The data has no records for that weekday. |
| `ModuleNotFoundError: magnetic_points` | Run `pip install -e .` from the repo root (or `export PYTHONPATH=src`). |
| Map shows grey tiles / no background | Folium loads OpenStreetMap tiles from the internet; open the HTML while online. |
| Matplotlib "no display" errors | Plots use the non-interactive Agg backend through the CLI; if you call functions in a script on a headless server set `MPLBACKEND=Agg`. |
| `evaluate` is slow | Random Forest × 10 folds on ~120k rows. Set `evaluation.n_jobs: -1` or `--models K-NN "Decision Tree"`. |
| `cluster-metrics` uses lots of memory/time | Lower `cluster_metrics.silhouette_sample_size`. |
| Different cluster numbering than the paper | See step 4 above. |
| Tests fail with `No module named haversine` | `pip install -r requirements-dev.txt`. |
