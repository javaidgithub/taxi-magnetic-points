"""Stage 5 - static plots and interactive Folium maps.

Ports the plotting cells of the model notebook (9, 12, 18, 19, 22-27 heat-maps,
33 overview map, 36 recommendation map).

Deliberate change: scatter plots put **longitude on the x-axis and latitude on
the y-axis**. The notebook's cluster-1 and combined-cluster scatter plots
(cells 12 and 18) used the opposite orientation, which the paper's Fig. 4/5
inherit.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from . import DAY_NAMES


def _plt():
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    return plt


def plot_cluster_scatter(df: pd.DataFrame, path: str, label: str = "Cluster",
                         title: str = "K-Means clusters (latitude / longitude)") -> str:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df["VehLng"], df["VehLat"], c=df[label], cmap="viridis", s=8)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_cluster_sizes(df: pd.DataFrame, path: str, label: str = "Cluster") -> str:
    plt = _plt()
    sizes = df[label].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([str(i) for i in sizes.index], sizes.values)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of pickups")
    ax.set_title("Data points per cluster")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_points_by_day(df: pd.DataFrame, path: str) -> str:
    plt = _plt()
    grouped = df.groupby(["DayOfWeek", "Cluster"]).size().unstack(fill_value=0)
    ax = grouped.plot(kind="bar", figsize=(9, 5))
    ax.set_title("Number of points per cluster by day of week")
    ax.set_xlabel("Day of week (0 = Monday)")
    ax.set_ylabel("Number of points")
    ax.legend(title="Cluster", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_confusion_matrix(cm: np.ndarray, labels: Sequence[int], title: str, path: str,
                          cmap: str = "Blues") -> str:
    import seaborn as sns
    plt = _plt()
    names = [f"Cluster-{i}" for i in labels]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, xticklabels=names, yticklabels=names, ax=ax,
                annot_kws={"size": 6})
    ax.set_title(title)
    ax.set_xlabel("Predicted Cluster")
    ax.set_ylabel("Actual Cluster")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def create_overview_map(df: pd.DataFrame, zoom_start: int = 12):
    """Folium map with one red marker per cluster and one blue marker per (cluster, day)."""
    import folium
    from folium.plugins import MarkerCluster

    m = folium.Map(location=[df["VehLat"].mean(), df["VehLng"].mean()], zoom_start=zoom_start)
    group = MarkerCluster().add_to(m)
    for cluster_id in sorted(df["Cluster"].unique()):
        cdata = df[df["Cluster"] == cluster_id]
        folium.Marker((cdata["VehLat"].mean(), cdata["VehLng"].mean()),
                      popup=f"Cluster {cluster_id}", icon=folium.Icon(color="red")).add_to(group)
        for day in range(7):
            sdata = cdata[cdata["DayOfWeek"] == day]
            if sdata.empty:
                continue
            folium.Marker((sdata["VehLat"].mean(), sdata["VehLng"].mean()),
                          popup=f"Subcluster (Cluster {cluster_id}, Day {day})<br>Points: {len(sdata)}",
                          icon=folium.Icon(color="blue")).add_to(group)
    folium.LayerControl().add_to(m)
    return m


def create_recommendation_map(recommendations: pd.DataFrame, driver_lat: float, driver_lng: float,
                              day_of_week: int, time_window: int = 10, zoom_start: int = 13):
    """Folium map: red marker = driver, blue markers = recommended magnetic points."""
    import folium

    m = folium.Map(location=[driver_lat, driver_lng], zoom_start=zoom_start)
    folium.Marker([driver_lat, driver_lng], popup="Driver Location",
                  icon=folium.Icon(color="red")).add_to(m)
    for _, row in recommendations.iterrows():
        folium.Marker(
            [row["VehLat"], row["VehLng"]],
            popup=(f"Cluster: {int(row['Cluster'])}<br>SubCluster: {int(row['SubCluster'])}<br>"
                   f"Day: {DAY_NAMES[day_of_week]}<br>Distance: {row['Distance_km']:.2f} km<br>"
                   f"Pickups (±{time_window} min): {int(row['Pickups_in_window'])}"),
            icon=folium.Icon(color="blue"),
        ).add_to(m)
    return m
