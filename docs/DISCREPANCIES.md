# Discrepancies between paper, research document and code

This repository treats the **source code as the authority for implementation
details** and the **paper as the authority for published claims**. Where they
disagree, nothing was silently corrected: each item is listed here with what
the repository does about it.

Sources compared: the uploaded notebooks (`notebooks/original/`), the camera-ready
paper PDF, and `Research.docx` (the authors' working description).

Status legend: **Fixed** = corrected in `src/` (original notebook left untouched) ·
**Flagged** = documented, behaviour unchanged · **Open** = needs an answer from the authors.

## A. Code vs code

| # | Finding | Evidence | Status |
|---|---|---|---|
| A1 | The preprocessing notebook computes `Minutes` with `dt.hour` (bug) and never drops `TourId`. The model notebook's input (`TripLog_DataSet.csv`) has real minute values and no `TourId`, so **the uploaded preprocessing notebook is not the code that produced the modelled data**. | PreProcessing cells 7-11 vs Model cells 1-2 | **Fixed** in `preprocessing.py` (`dt.minute`, `TourId` dropped). Test: `test_minutes_are_minutes_not_hours`. **Open**: please confirm which version produced the paper's data. |
| A2 | Pickups are selected by *dropping* `Status == 8`, not by *keeping* `Status == 6`. Equivalent only if no other status codes exist. | PreProcessing cell 3 | **Fixed** (keeps `Status == 6`, as the paper says) and a warning is raised if other codes appear. **Open**: do other status codes exist in the raw file? |
| A3 | The recursive step hard-codes cluster `0`. It happens to be the largest cluster (101,017 points). | Model cell 10 | **Fixed**: `split_cluster: auto` picks the largest; `split_cluster: 0` reproduces the notebook. |
| A4 | Scatter plots in cells 12 and 18 put latitude on the x-axis (axis labels agree with this, but the orientation is opposite to cell 9). The paper's Fig. 4 and Fig. 5 inherit it. | Model cells 9, 12, 18 | **Fixed** in `visualization.py` (longitude = x). Original figures kept unchanged in `docs/figures/original/`. |
| A5 | Two prediction implementations exist. Cell 35 (older, ~6 s) matches on the exact hour/minute/day and returns the nearest *individual records*; cell 36 (newer, 0.4733 s) uses per-cluster-per-day *centroids* and a ±window pickup count. The paper's 0.4733 s corresponds to cell 36. | Model cells 35-36 | **Flagged**. Only cell 36 is ported. |
| A6 | Complexity labels printed by cell 35 ("O(n log n)", "O(1)") sit on `iterrows` loops, which are O(n) in rows scanned. | Model cell 35 | **Flagged**. Not carried over. |
| A7 | `SubCluster` is stored as an object column (`None` then ints). | Model cell 28 | **Fixed**: integer column. |
| A8 | The final map popup hard-codes "±10 min" even if `time_window` changes. | Model cell 36 | **Fixed**: uses the actual window. |

## B. Paper vs code

| # | Paper / docx says | Code shows | Status |
|---|---|---|---|
| B1 | 121,254 pickups ("half of 242,508"). | 242,509 raw rows → **121,422** pickups. The paper's own Fig. 6 bars sum to 121,422. | **Flagged**. README uses the notebook figures. |
| B2 | Table II/IV show timestamps `1/1/2025` with `DayOfWeek = 0`. | Data starts `2024-01-01` (a Monday). 1 Jan 2025 was a Wednesday, so the paper's table is internally inconsistent. | **Flagged**. |
| B3 | Classifiers use "VehLat, VehLng, Cluster" as features. | Features are `VehLat, VehLng`; `Cluster` is the **label**. Using the label as a feature would be leakage. | **Flagged**; repository documents the code's behaviour. |
| B4 | "The trained K-NN model was selected for deployment" and predicts the top-3 sub-clusters. | No model is trained for prediction or saved (no `joblib`/`pickle`). Prediction is nearest-centroid by haversine distance. | **Flagged**. README does not claim a deployed KNN. |
| B5 | Time (hour/minute) is part of the "spatio-temporal context" for the recommendation. | Time only changes the pickup *count* in the window; the 3 locations depend on day and position only. | **Flagged**; see `docs/METHODOLOGY.md`. |
| B6 | "7 day-wise **sub-clusters**" / "133 sub-clusters". | `SubCluster = DayOfWeek`; no clustering is run. Whether all 133 (Cluster × day) pairs are non-empty cannot be checked from the saved outputs (cell 31 output is truncated). | **Flagged**. |
| B7 | Silhouette 0.736, Davies–Bouldin 0.320, Calinski–Harabasz 2,556,892. | **No notebook computes any of them.** Feature set, scaling and sampling are unknown. | **Open**. `evaluation.clustering_metrics` / `cluster-metrics` command added as *new* code to let authorized users compute them; values are not guaranteed to match. |
| B8 | Prediction example: 0.26 / 3.04 / 6.45 km with 40 / 24 / 12 pickups. | Not present in any saved cell output (only the map object). | **Open** (cannot verify without the data). |
| B9 | "The time complexity … was computed to be 0.4733 seconds." | That is a wall-clock runtime on Colab (cell 36 output), not a complexity class. | **Flagged**; README reports it as a runtime. |
| B10 | Paper Fig. 2/4/5 naming "Cluster 1" is the large cluster. | Notebook label is `0` (paper uses 1-based names in Figs. 3 and 6). | **Flagged** (naming only). |
| B11 | Paper names the dataset attribute `VehId` (text and Table II) and also writes "reatedDate" / "AbsouIT" / "Minute". | Code and docx use `TourId`, `CreatedDate`, `Minutes`; vendor site is absoluit.com. | **Flagged**; repository uses the code's names. |
| B12 | "Pickups ... Half of those points were pickup points". | The split is not exactly half (121,422 / 242,509). | **Flagged**. |

## C. Research document (`Research.docx`) vs paper

| # | `Research.docx` | Paper / code | Status |
|---|---|---|---|
| C1 | 20 spatial clusters, 140 sub-clusters (20 × 7). | 19 clusters, 133 segments; the code produces 19. | **Flagged**. README uses 19 / 133. |
| C2 | Dataset has 242,509 entries. | Paper says 242,508; notebook counts 242,509. | **Flagged**. |

## D. Things the repository could not verify

* The IEEE Xplore page (`https://ieeexplore.ieee.org/document/11591654`) could not be read by the tool used, and the camera-ready PDF carries IEEE's template placeholders instead of venue, year or DOI. **Publication venue, year, volume/pages and DOI are therefore left as `TODO` placeholders.**
* The AbsoluIT dataset was not provided, so **none of the paper's numeric results were reproduced here**. The classifier table and cluster sizes were cross-checked only against the *saved outputs inside the uploaded notebook* (they agree).
* scikit-learn version used on Colab is not recorded in the notebook. K-Means' default `n_init` changed in scikit-learn 1.4, so exact label numbering may differ between versions even with `random_state=42`. Use `clustering.n_init` in `configs/default.yaml` to pin it.

## E. Methodological caveat (not an error)

The cluster labels are generated by K-Means on `VehLat`/`VehLng`, and the classifiers are then trained to predict those labels from the same two features. The ≈99.8 % accuracy shows the partition is easy to learn (K-Means regions are convex cells of the coordinate space); it does **not** validate demand prediction, time-awareness, or recommendation quality on held-out future data.
