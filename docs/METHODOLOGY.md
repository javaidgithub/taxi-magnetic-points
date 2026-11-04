# Methodology, concept by concept

Each concept follows the same template: **What · Why · How · In this project · Output · Next stage.**
Everything below describes what the *code* does; where the paper words it differently
see [DISCREPANCIES.md](DISCREPANCIES.md).

---

## 1. Datetime feature extraction

* **What** – Turning one timestamp (`CreatedDate`, e.g. `2024-01-01 00:01`) into numbers a model can use.
* **Why** – Clustering and classifiers need numeric inputs, and taxi demand depends on time of day and day of week.
* **How** – `pandas.to_datetime` then `.dt.hour` (0–23), `.dt.minute` (0–59), `.dt.day_of_week` (0 = Monday … 6 = Sunday).
* **In this project** – `preprocessing.add_datetime_features`. The column is named `Minutes` (as in the code), not `Minute` (as in the paper).
* **Output** – three integer columns.
* **Next** – `DayOfWeek` becomes the day-wise sub-cluster; `Hour`/`Minutes` are used by the predictor's time window. They are **not** used for K-Means.

## 2. Pickup filtering (`Status`)

* **What** – Each log row is either a pickup (`Status = 6`) or a drop-off (`Status = 8`).
* **Why** – Hotspots for drivers are places where passengers *board*, so drop-offs are removed.
* **How** – keep rows with `Status == 6`.
* **In this project** – `preprocessing.filter_pickups`; warns if other codes appear.
* **Output** – pickup-only table. On the paper's data: 242,509 → 121,422 rows (notebook output).
* **Next** – the table is saved as `TripLog_DataSet.csv` with columns `VehLat, VehLng, Hour, Minutes, DayOfWeek`.

## 3. K-Means clustering

* **What** – An unsupervised algorithm that groups points into *k* clusters by repeatedly (a) assigning each point to its nearest centre and (b) moving each centre to the mean of its points.
* **Why** – To discover pickup regions without labels.
* **How** – scikit-learn `KMeans(n_clusters=10, random_state=42)` on `VehLat`, `VehLng` (raw degrees, no scaling, Euclidean distance).
* **In this project** – `clustering.initial_kmeans`.
* **Output** – a label 0–9 per pickup. On the paper's data one cluster held 101,017 of 121,422 points, because pickups are heavily concentrated in the city core while a few far-away groups (other cities) each attract a cluster.
* **Next** – the imbalance motivates the recursive step.

## 4. Recursive clustering (hierarchical K-Means)

* **What** – Applying K-Means again *inside* an oversized cluster.
* **Why** – One dominant cluster hides the structure of the dense core.
* **How** – take the points of the largest cluster, run `KMeans(n_clusters=10)` on them, and give the results labels `0..9 + 10` so they cannot clash with the untouched clusters.
* **In this project** – `clustering.refine_cluster`. Exactly one level of recursion is implemented (one split), giving 9 + 10 = 19 clusters. The notebook hard-coded cluster 0; the library picks the largest automatically (`split_cluster: auto`).
* **Output** – 19 spatial clusters (paper's data: sizes from 88 to 36,696 points).
* **Next** – each cluster is split by day.

## 5. Day-wise sub-clustering

* **What** – Giving every pickup a `SubCluster` id equal to its day of week.
* **Why** – Demand differs by weekday, so a hotspot on Friday is treated separately from the same region on Monday.
* **How** – `SubCluster = DayOfWeek`. **No clustering algorithm runs in this step**; (Cluster, SubCluster) pairs (19 × 7 = 133) are simply groups.
* **In this project** – `clustering.assign_day_subcluster`; `clustering.segment_counts` tabulates the groups.
* **Output** – `TripLog_Clustered.csv` with `Cluster` and `SubCluster`.
* **Next** – classification and prediction both use these groups.

## 6. Cluster-quality metrics (Silhouette, Davies–Bouldin, Calinski–Harabasz)

* **What** – Scores for how compact and separated clusters are. Silhouette ∈ [−1, 1], higher is better; Davies–Bouldin ≥ 0, lower is better; Calinski–Harabasz ≥ 0, higher is better.
* **Why** – To judge the clustering without labels.
* **How** – scikit-learn `silhouette_score`, `davies_bouldin_score`, `calinski_harabasz_score` on the coordinates and final `Cluster` labels.
* **In this project** – `evaluation.clustering_metrics` and the `cluster-metrics` command. **This is new code**: the paper reports 0.736 / 0.320 / 2,556,892 but none of the uploaded notebooks compute them. Silhouette is computed on a random sample (default 20,000 points) because it needs all pairwise distances.
* **Output** – `outputs/cluster_metrics.json`.
* **Next** – none (reporting).

## 7. Supervised classification (KNN, Decision Tree, Random Forest)

* **What** – Models that learn to predict a label from features.
  * *KNN (k = 3)*: majority vote of the 3 nearest training points.
  * *Decision Tree*: a sequence of threshold questions on the features.
  * *Random Forest (100 trees)*: many randomised trees voting together.
* **Why** – To check that the cluster partition is *learnable*: if a model can recover the cluster from coordinates, regions are consistent and well separated.
* **How** – features `VehLat`, `VehLng`; label `Cluster` (19 classes).
* **In this project** – `evaluation.build_classifiers`, `evaluation.evaluate_classifiers`.
* **Output** – a metrics table and a confusion matrix per model.
* **Next** – reporting only. **No classifier is used by the predictor** (see DISCREPANCIES B4).

## 8. 10-fold stratified cross-validation

* **What** – Split the data into 10 folds that preserve class proportions; train on 9, test on 1, repeat 10 times.
* **Why** – Every point is tested once and class imbalance (88 vs 36,696 points) does not leave a class out of a fold.
* **How** – `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`; Accuracy plus **weighted** Precision/Recall/F1 averaged over folds; predictions from all folds are pooled for the confusion matrix.
* **In this project** – `evaluation.cross_validate_classifier` (verified equal to the notebook's `cross_val_score` calls by `tests/test_evaluation.py`).
* **Output** – per-classifier means (paper's data, notebook output): KNN 0.9979, Decision Tree 0.9975, Random Forest 0.9983 for all four metrics.
* **Caveat** – labels come from K-Means on the same features, so high scores are expected; see DISCREPANCIES §E.

## 9. Magnetic-point prediction

* **What** – Given the driver's position, day and time, return the three nearest pickup hotspots and how busy they are.
* **Why** – To help a driver who does not know local demand patterns.
* **How** – (1) keep the requested day; (2) centroid = mean lat/lng of each cluster's pickups on that day; (3) haversine distance driver → each centroid; (4) take the 3 nearest; (5) count each cluster-day's pickups whose time of day lies within ± `time_window` minutes of the requested time (default 10, no midnight wrap).
* **In this project** – `prediction.MagneticPointPredictor`. The constructor pre-computes centroids and sorted time arrays so each query is fast; results are tested equal to a literal port of the notebook function (`tests/test_prediction.py`).
* **Output** – a table `Cluster, SubCluster, VehLat, VehLng, Distance_km, Pickups_in_window`.
* **Limitations** – centroids average *all* hours of the day, so a point can be nearest even if it is quiet at the requested time; a centroid of a large, spread-out cluster may lie in a low-demand location.

## 10. Geographic visualisation (Folium)

* **What** – Interactive Leaflet maps rendered to HTML.
* **Why** – Drivers/analysts can pan, zoom and click markers.
* **How** – `folium.Map`, red marker = driver, blue markers = recommended points with a popup (cluster, day, distance, pickups). An overview map shows one red marker per cluster and one blue marker per cluster-day inside a `MarkerCluster`.
* **In this project** – `visualization.create_recommendation_map`, `visualization.create_overview_map`.
* **Output** – `outputs/nearest_subclusters_map.html`, `outputs/clusters_map.html`. Map tiles load from OpenStreetMap, so viewing needs internet access.
