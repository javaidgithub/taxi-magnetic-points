"""Regenerate docs/figures/sample/* from the SYNTHETIC sample data.

Usage:  python scripts/make_doc_figures.py
All figures produced here come from synthetic data (data/sample/TripLog_sample.csv)
and illustrate the pipeline only; they are NOT results from the paper.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from magnetic_points import clustering, evaluation, preprocessing
from magnetic_points import visualization as viz
from magnetic_points.prediction import MagneticPointPredictor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures" / "sample"
OUT.mkdir(parents=True, exist_ok=True)

raw = pd.read_csv(ROOT / "data/sample/TripLog_sample.csv")
pre = preprocessing.preprocess(raw)
lab0 = clustering.initial_kmeans(pre)
first = pre.assign(Cluster=lab0)
final = clustering.run_clustering(pre)

viz.plot_cluster_scatter(first, str(OUT / "sample_initial_10_clusters.png"), title="Synthetic sample - initial 10 K-Means clusters")
viz.plot_cluster_scatter(final, str(OUT / "sample_final_19_clusters.png"), title="Synthetic sample - final 19 clusters")
viz.plot_cluster_sizes(final, str(OUT / "sample_cluster_sizes.png"))
res, cm, labels = evaluation.evaluate_classifiers(final, models=["K-NN"])
viz.plot_confusion_matrix(cm["K-NN"], labels, "Synthetic sample - K-NN confusion matrix (10-fold CV)", str(OUT / "sample_knn_confusion_matrix.png"))
viz.create_overview_map(final).save(str(OUT / "sample_clusters_map.html"))

# Example recommendation: Friday 22:30, +/-30 min window
lat, lng, hour, minute, day, window = 59.909851, 10.743474, 22, 30, 4, 30
rec = MagneticPointPredictor(final).predict(lat, lng, hour, minute, day, 3, window)
rec.round(4).to_csv(OUT / "sample_recommendation.csv", index=False)
viz.create_recommendation_map(rec, lat, lng, day, window).save(str(OUT / "sample_recommendation_map.html"))

# static preview of the recommendation (the Folium map itself is interactive HTML)
sub = final[(final["VehLat"].between(59.88, 59.96)) & (final["VehLng"].between(10.66, 10.84))]
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(sub["VehLng"], sub["VehLat"], c="lightgray", s=6, label="pickups (synthetic)")
ax.scatter([lng], [lat], c="red", marker="*", s=250, label="driver", zorder=3)
for i, r in rec.iterrows():
    ax.scatter([r["VehLng"]], [r["VehLat"]], c="tab:blue", s=120, zorder=3)
    ax.annotate(f"#{i+1}  cluster {int(r['Cluster'])}\n{r['Distance_km']:.2f} km, {int(r['Pickups_in_window'])} pickups",
                (r["VehLng"], r["VehLat"]), textcoords="offset points", xytext=(8, 8), fontsize=8)
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.set_title("Synthetic sample - top-3 magnetic points (Fri 22:30, +/-30 min)")
ax.legend(loc="lower left"); fig.tight_layout()
fig.savefig(OUT / "sample_recommendation_preview.png", dpi=150)
print(res.round(4)); print(rec.round(4))
