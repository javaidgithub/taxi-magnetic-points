"""Command-line interface.

Run ``python -m magnetic_points --help`` (after ``pip install -e .``) or see
README.md for the full list of commands.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import DAY_NAMES, __version__
from .config import load_config


def _mkdir_for(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------- commands
def cmd_make_sample(args, cfg):
    from .synthetic import generate_trip_log
    df = generate_trip_log(n_trips=args.n_trips, seed=args.seed)
    _mkdir_for(args.out)
    df.to_csv(args.out, index=False)
    print(f"Wrote SYNTHETIC trip log with {len(df)} rows -> {args.out}")


def cmd_preprocess(args, cfg):
    from . import preprocessing as pp
    inp = args.input or cfg["paths"]["raw_csv"]
    out = args.output or cfg["paths"]["preprocessed_csv"]
    raw = pp.load_raw(inp)
    processed = pp.preprocess(raw, cfg["preprocessing"]["pickup_status"],
                              cfg["preprocessing"]["datetime_column"])
    _mkdir_for(out)
    processed.to_csv(out, index=False)
    print(f"Raw rows: {len(raw)} | pickup rows: {len(processed)} -> {out}")


def cmd_cluster(args, cfg):
    from . import clustering as cl
    inp = args.input or cfg["paths"]["preprocessed_csv"]
    out = args.output or cfg["paths"]["clustered_csv"]
    c = cfg["clustering"]
    df = pd.read_csv(inp)
    result = cl.run_clustering(df, c["n_initial_clusters"], c["n_sub_clusters"],
                               c["split_cluster"], c["random_state"], c["features"], c.get("n_init"))
    _mkdir_for(out)
    result.to_csv(out, index=False)
    sizes = cl.cluster_sizes(result)
    print(f"{len(sizes)} clusters; sizes:\n{sizes.to_string()}\n-> {out}")


def cmd_cluster_metrics(args, cfg):
    from . import evaluation as ev
    inp = args.input or cfg["paths"]["clustered_csv"]
    m = cfg["cluster_metrics"]
    df = pd.read_csv(inp)
    res = ev.clustering_metrics(df, cfg["clustering"]["features"], "Cluster",
                                m["silhouette_sample_size"], m["random_state"])
    print(json.dumps(res, indent=2))
    out_dir = Path(args.out_dir or cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cluster_metrics.json").write_text(json.dumps(res, indent=2))


def cmd_evaluate(args, cfg):
    from . import evaluation as ev, visualization as viz
    inp = args.input or cfg["paths"]["clustered_csv"]
    e = cfg["evaluation"]
    out_dir = Path(args.out_dir or cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(inp)
    results, confusion, labels = ev.evaluate_classifiers(
        df, e["features"], e["label"], e["n_splits"], e["random_state"],
        e["knn_neighbors"], e["random_forest_trees"], e["n_jobs"], args.models)
    print(results.round(4).to_string())
    results.round(4).to_csv(out_dir / "classifier_results.csv")
    cmaps = {"K-NN": "Blues", "Decision Tree": "Greens", "Random Forest": "Oranges"}
    for name, cm in confusion.items():
        fname = name.lower().replace(" ", "_").replace("-", "") + "_confusion_matrix.png"
        viz.plot_confusion_matrix(cm, labels, f"{name} - Confusion Matrix ({e['n_splits']}-Fold CV)",
                                  str(out_dir / fname), cmaps.get(name, "Blues"))
    print(f"Results and confusion matrices -> {out_dir}")


def cmd_plots(args, cfg):
    from . import visualization as viz
    inp = args.input or cfg["paths"]["clustered_csv"]
    out_dir = Path(args.out_dir or cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(inp)
    viz.plot_cluster_scatter(df, str(out_dir / "clusters_scatter.png"))
    viz.plot_cluster_sizes(df, str(out_dir / "cluster_sizes.png"))
    viz.plot_points_by_day(df, str(out_dir / "points_by_day.png"))
    viz.create_overview_map(df, cfg["visualization"]["map_zoom_overview"]).save(str(out_dir / "clusters_map.html"))
    print(f"Plots and overview map -> {out_dir}")


def cmd_predict(args, cfg):
    from . import prediction as pr, visualization as viz
    inp = args.input or cfg["paths"]["clustered_csv"]
    p = cfg["prediction"]
    k = args.num_neighbors or p["num_neighbors"]
    window = args.time_window if args.time_window is not None else p["time_window_minutes"]
    df = pd.read_csv(inp)
    predictor = pr.MagneticPointPredictor(df)
    res = predictor.predict(args.lat, args.lng, args.hour, args.minute, args.day, k, window)
    print(f"Driver at ({args.lat}, {args.lng}) on {DAY_NAMES[args.day]} {args.hour:02d}:{args.minute:02d}")
    print(res.round(4).to_string(index=False))
    print(f"Elapsed: {res.attrs['elapsed_seconds']:.4f} s (excludes loading the CSV)")
    out_map = args.out_map or str(Path(cfg["paths"]["output_dir"]) / "nearest_subclusters_map.html")
    _mkdir_for(out_map)
    viz.create_recommendation_map(res, args.lat, args.lng, args.day, window,
                                  cfg["visualization"]["map_zoom_recommendation"]).save(out_map)
    print(f"Map -> {out_map}")


def cmd_run_all(args, cfg):
    """preprocess -> cluster -> evaluate -> cluster-metrics -> plots -> example prediction."""
    ns = argparse.Namespace
    cmd_preprocess(ns(input=args.raw, output=None), cfg)
    cmd_cluster(ns(input=None, output=None), cfg)
    cmd_evaluate(ns(input=None, out_dir=args.out_dir, models=None), cfg)
    cmd_cluster_metrics(ns(input=None, out_dir=args.out_dir), cfg)
    cmd_plots(ns(input=None, out_dir=args.out_dir), cfg)
    cmd_predict(ns(input=None, lat=59.909851, lng=10.743474, hour=14, minute=30, day=1,
                   num_neighbors=None, time_window=None,
                   out_map=str(Path(args.out_dir or cfg["paths"]["output_dir"]) / "nearest_subclusters_map.html")), cfg)


# ----------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="magnetic_points", description=__doc__)
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("--config", help="YAML config overriding configs/default.yaml values")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("make-sample", help="generate a SYNTHETIC trip log (same schema as the raw data)")
    s.add_argument("--out", default="data/sample/TripLog_sample.csv")
    s.add_argument("--n-trips", type=int, default=7000)
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(func=cmd_make_sample)

    s = sub.add_parser("preprocess", help="raw log -> pickup table with Hour/Minutes/DayOfWeek")
    s.add_argument("--input"); s.add_argument("--output")
    s.set_defaults(func=cmd_preprocess)

    s = sub.add_parser("cluster", help="recursive K-Means + day-wise SubCluster labels")
    s.add_argument("--input"); s.add_argument("--output")
    s.set_defaults(func=cmd_cluster)

    s = sub.add_parser("evaluate", help="10-fold CV of KNN / Decision Tree / Random Forest")
    s.add_argument("--input"); s.add_argument("--out-dir")
    s.add_argument("--models", nargs="+", choices=["K-NN", "Decision Tree", "Random Forest"])
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("cluster-metrics", help="Silhouette / Davies-Bouldin / Calinski-Harabasz (new code)")
    s.add_argument("--input"); s.add_argument("--out-dir")
    s.set_defaults(func=cmd_cluster_metrics)

    s = sub.add_parser("plots", help="scatter, size, per-day plots and overview map")
    s.add_argument("--input"); s.add_argument("--out-dir")
    s.set_defaults(func=cmd_plots)

    s = sub.add_parser("predict", help="recommend the nearest magnetic points for a driver")
    s.add_argument("--input")
    s.add_argument("--lat", type=float, required=True)
    s.add_argument("--lng", type=float, required=True)
    s.add_argument("--hour", type=int, required=True)
    s.add_argument("--minute", type=int, default=0)
    s.add_argument("--day", type=int, required=True, help="0=Monday ... 6=Sunday")
    s.add_argument("--num-neighbors", type=int)
    s.add_argument("--time-window", type=int, help="+/- minutes used to count pickups")
    s.add_argument("--out-map")
    s.set_defaults(func=cmd_predict)

    s = sub.add_parser("run-all", help="run the entire pipeline")
    s.add_argument("--raw", help="raw CSV (default: paths.raw_csv)")
    s.add_argument("--out-dir")
    s.set_defaults(func=cmd_run_all)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    try:
        args.func(args, cfg)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
