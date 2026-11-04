# Developer guide

Install first: see the README ("Installation"). All snippets assume the package is installed (`pip install -e .`).

## 1. Change clustering parameters

Edit `configs/default.yaml` (or copy it and pass `--config my.yaml`; only changed keys are needed):

```yaml
clustering:
  n_initial_clusters: 12   # first K-Means
  n_sub_clusters: 8        # split of the largest cluster
  split_cluster: auto      # or an integer label
  n_init: 10               # pin K-Means restarts for cross-version stability
```

```bash
python -m magnetic_points --config my.yaml cluster
```

The final number of clusters is `n_initial_clusters − 1 + n_sub_clusters`. Downstream code
reads the labels dynamically, so confusion-matrix sizes adapt. Note: the **paper's
numbers (19 clusters) correspond to 10 and 10**.

From Python:

```python
import pandas as pd
from magnetic_points.clustering import run_clustering
df = pd.read_csv("data/processed/TripLog_DataSet.csv")
clustered = run_clustering(df, n_initial_clusters=12, n_sub_clusters=8)
```

To recurse more than once, call `refine_cluster` again on the next-largest cluster
(use a `label_offset` larger than any existing label, e.g. `labels.max() + 1`).

## 2. Replace or add a classifier

Edit `evaluation.build_classifiers`:

```python
from sklearn.ensemble import GradientBoostingClassifier

def build_classifiers(...):
    return {
        "K-NN": KNeighborsClassifier(n_neighbors=knn_neighbors),
        ...
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
    }
```

Then add the name to the `--models` choices in `cli.py` (and a colormap in `cmd_evaluate`
if you want a custom heat-map colour). Any scikit-learn estimator with `fit`/`predict` works.

Change hyper-parameters (e.g. `k` for KNN) in `configs/default.yaml → evaluation`.

## 3. Use another dataset

Provide a CSV with the **raw schema**:

| Column | Meaning |
|---|---|
| `TourId` | trip identifier (dropped) |
| `VehLat`, `VehLng` | WGS-84 coordinates in decimal degrees |
| `Status` | `6` pickup, `8` drop-off (other codes are dropped with a warning) |
| `CreatedDate` | anything `pandas.to_datetime` can parse |

```bash
python -m magnetic_points run-all --raw path/to/your.csv
```

If your pickup code is not 6, set `preprocessing.pickup_status`. If your timestamp column
has another name, set `preprocessing.datetime_column`. If you already have a pickup-only
table with `VehLat, VehLng, Hour, Minutes, DayOfWeek`, skip `preprocess` and put the file at
`paths.preprocessed_csv`.

Things to check on new data: raw degrees are clustered with Euclidean distance (fine for a
single city; for a wide area consider projecting coordinates); far-away outliers can take
whole clusters; every cluster needs ≥ `n_splits` (10) members for clean stratified CV (a
warning is suppressed inside the CV loop, so check `cluster_sizes`).

## 4. Change prediction inputs

CLI:

```bash
python -m magnetic_points predict --lat 59.91 --lng 10.74 --hour 22 --minute 30 --day 4 \
    --num-neighbors 5 --time-window 30 --out-map outputs/my_map.html
```

Python:

```python
from magnetic_points.prediction import MagneticPointPredictor
p = MagneticPointPredictor(clustered)                 # build once ...
p.predict(59.91, 10.74, 22, 30, 4, num_neighbors=5, time_window=30)   # ... query many times
```

Defaults (`num_neighbors`, `time_window_minutes`) live in `configs/default.yaml → prediction`.
Day is 0 = Monday … 6 = Sunday. Invalid values raise `ValueError`.

To make the *time* influence which points are chosen (currently it only affects counts),
extend `MagneticPointPredictor.__init__` to compute centroids per (day, hour) or per
time-window instead of per day; add a test next to `test_matches_notebook_implementation`
that documents the new behaviour (the existing test pins the notebook's behaviour).

## 5. Extend the visualisation system

* New static plot: add a function to `visualization.py` that takes a DataFrame and a path, and call it from `cmd_plots` in `cli.py`.
* Richer map: `create_recommendation_map` returns a normal `folium.Map`, so you can add layers, e.g. a heat map:

```python
from folium.plugins import HeatMap
m = create_recommendation_map(rec, lat, lng, day)
HeatMap(clustered[["VehLat", "VehLng"]].values.tolist(), radius=8).add_to(m)
m.save("outputs/heat.html")
```

* Real-time or service use: wrap `MagneticPointPredictor.predict` in a web framework; build the predictor once at start-up.

## 6. Coding conventions

* Keep notebook behaviour pinned by tests; if you intentionally change behaviour, update `docs/DISCREPANCIES.md` and `CHANGELOG.md`.
* Run `python -m pytest` before opening a pull request.
