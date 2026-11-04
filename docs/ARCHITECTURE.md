# Architecture and workflow diagrams

All diagrams are Mermaid (rendered natively by GitHub) and reflect the code in
`src/magnetic_points/`, not an idealised design.

## 1. System architecture

```mermaid
flowchart LR
    RAW[("Raw trip log<br/>TripLog.csv<br/>proprietary, not in repo")] --> PRE
    subgraph SRC["src/magnetic_points"]
        PRE["preprocessing.py<br/>filter pickups, Hour/Minutes/DayOfWeek"]
        CLU["clustering.py<br/>K-Means, recursive split, day label"]
        EVA["evaluation.py<br/>10-fold CV of KNN / DT / RF<br/>cluster-quality metrics"]
        PRD["prediction.py<br/>MagneticPointPredictor"]
        VIZ["visualization.py<br/>plots + Folium maps"]
    end
    CFG["configs/default.yaml"] -.-> PRE
    CFG -.-> CLU
    CFG -.-> EVA
    CFG -.-> PRD
    PRE --> D1[("TripLog_DataSet.csv")]
    D1 --> CLU
    CLU --> D2[("TripLog_Clustered.csv")]
    D2 --> EVA
    D2 --> PRD
    D2 --> VIZ
    EVA --> O1["classifier_results.csv<br/>confusion matrices<br/>cluster_metrics.json"]
    PRD --> O2["top-3 magnetic points"]
    O2 --> VIZ
    VIZ --> O3["PNG plots + HTML maps"]
    CLI["cli.py<br/>python -m magnetic_points"] --> PRE
    CLI --> CLU
    CLI --> EVA
    CLI --> PRD
    CLI --> VIZ
```

## 2. End-to-end workflow

```mermaid
flowchart TD
    A["Raw log: TourId, VehLat, VehLng, Status, CreatedDate"] --> B["Keep Status = 6 (pickups)"]
    B --> C["Extract Hour, Minutes, DayOfWeek;<br/>drop TourId, Status, CreatedDate"]
    C --> D["K-Means k=10 on VehLat, VehLng"]
    D --> E["Re-cluster the largest cluster<br/>with K-Means k=10, labels +10"]
    E --> F["19 spatial clusters"]
    F --> G["SubCluster = DayOfWeek<br/>(19 x 7 = 133 segments)"]
    G --> H["Clustered table"]
    H --> I["Stratified 10-fold CV:<br/>KNN, Decision Tree, Random Forest"]
    H --> J["Predictor: nearest 3 cluster-day centroids"]
    J --> K["Count pickups within +/- window minutes"]
    K --> L["Folium map"]
```

## 3. Data preprocessing workflow

```mermaid
flowchart LR
    R["load_raw()<br/>checks 5 columns"] --> S{"Status values<br/>other than 6 / 8?"}
    S -- yes --> W["warn"]
    S -- no --> F
    W --> F["filter_pickups()<br/>Status == 6"]
    F --> T["add_datetime_features()<br/>pd.to_datetime(CreatedDate)"]
    T --> H["Hour = .dt.hour"]
    T --> M["Minutes = .dt.minute"]
    T --> DW["DayOfWeek = .dt.day_of_week<br/>0 = Monday"]
    H --> OUT["select columns:<br/>VehLat, VehLng, Hour, Minutes, DayOfWeek"]
    M --> OUT
    DW --> OUT
```

## 4. Hierarchical (recursive) clustering

```mermaid
flowchart TD
    ALL["All pickups<br/>121,422 in the paper's data"] -->|"KMeans k=10<br/>VehLat, VehLng"| L0["10 clusters, labels 0-9"]
    L0 --> BIG["Largest cluster<br/>101,017 points (label 0)"]
    L0 --> REST["9 other clusters<br/>unchanged: labels 1-9"]
    BIG -->|"KMeans k=10<br/>labels + 10"| SUB["10 sub-clusters<br/>labels 10-19"]
    REST --> FIN["Final: 19 spatial clusters"]
    SUB --> FIN
    FIN -->|"SubCluster = DayOfWeek"| SEG["133 cluster x day segments"]
```

The split cluster is the *largest* one; on the paper's data that was label 0.
On other data it can be a different label (see `clustering.split_cluster`).

## 5. Model training and evaluation workflow

```mermaid
flowchart TD
    DF["Clustered table"] --> XY["X = VehLat, VehLng<br/>y = Cluster"]
    XY --> SK["StratifiedKFold<br/>10 splits, shuffle, seed 42"]
    SK --> FIT["for each fold: fit classifier, predict held-out fold"]
    FIT --> MET["Accuracy + weighted Precision / Recall / F1<br/>averaged over folds"]
    FIT --> POOL["pool out-of-fold predictions"]
    POOL --> CM["confusion matrix per classifier"]
    DF --> CQ["clustering_metrics()<br/>Silhouette (sampled), Davies-Bouldin, Calinski-Harabasz<br/>NEW code"]
    MET --> RES["classifier_results.csv"]
    CM --> PNG["*_confusion_matrix.png"]
    CQ --> JSON["cluster_metrics.json"]
```

## 6. Prediction pipeline

```mermaid
sequenceDiagram
    participant D as Driver input
    participant P as MagneticPointPredictor
    participant V as visualization
    Note over P: __init__ precomputes per (day, cluster) centroids<br/>and sorted minute-of-day arrays
    D->>P: lat, lng, hour, minute, day, k, window
    P->>P: validate ranges
    P->>P: select centroids of the requested day
    P->>P: haversine distance to every centroid
    P->>P: take k nearest (stable order)
    P->>P: count pickups in minute-of-day +/- window
    P-->>D: DataFrame Cluster, SubCluster, VehLat, VehLng, Distance_km, Pickups_in_window
    D->>V: create_recommendation_map()
    V-->>D: Folium HTML map
```

The requested time only affects the pickup counts; it does not change which three
locations are chosen (see `docs/DISCREPANCIES.md` B5).
