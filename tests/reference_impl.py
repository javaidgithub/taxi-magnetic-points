"""Literal ports of the ORIGINAL notebook code, used as test oracles.

These intentionally keep the notebook's slow ``iterrows`` structure so the
refactored, vectorised code can be compared against it.
"""
from datetime import timedelta

from haversine import haversine


def notebook_predict(df, latitude, longitude, hour, minute, day_of_week, num_neighbors=3, time_window=10):
    """Notebook cell 36 ``predict_nearest_subclusters`` (timing prints removed)."""
    day_df = df[df["DayOfWeek"] == day_of_week]
    centroids = (day_df.groupby(["Cluster", "SubCluster"])[["VehLat", "VehLng"]].mean().reset_index())
    centroids["Distance"] = centroids.apply(
        lambda r: haversine((latitude, longitude), (r["VehLat"], r["VehLng"])), axis=1)
    nearest = centroids.nsmallest(num_neighbors, "Distance")
    points_info = {}
    driver_time = timedelta(hours=hour, minutes=minute)
    window = timedelta(minutes=time_window)
    for _, row in nearest.iterrows():
        sub_df = day_df[(day_df["Cluster"] == row["Cluster"]) & (day_df["SubCluster"] == row["SubCluster"])]
        count = 0
        for _, rec in sub_df.iterrows():
            if abs(timedelta(hours=int(rec["Hour"]), minutes=int(rec["Minutes"])) - driver_time) <= window:
                count += 1
        points_info[(row["Cluster"], row["SubCluster"])] = count
    return nearest, points_info
