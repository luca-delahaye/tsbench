"""Command line: run the whole thing on a CSV.

    python -m tsbench run data/day.csv --time dteday --value cnt --horizon 7

Decides nothing. Parses flags, calls the pieces in order, prints the table.
"""

import argparse

import numpy as np
import pandas as pd

from tsbench.data import load_series
from tsbench.evaluate import aggregate, evaluate
from tsbench.models import Drift, Mean, Naive, SeasonalNaive


def default_period(freq):
    """The obvious seasonal cycle for this spacing, as a count of positions."""
    known = {pd.Timedelta(days=1): 7, pd.Timedelta(hours=1): 24}
    return known.get(freq)


def describe_freq(freq):
    """Readable spacing, so a header says 'daily' not '1 days 00:00:00'."""
    known = {pd.Timedelta(days=1): "daily", pd.Timedelta(hours=1): "hourly",
             pd.Timedelta(weeks=1): "weekly", pd.Timedelta(minutes=1): "per minute"}
    return known.get(freq, str(freq))


def build_models(period):
    """The baselines, plus seasonal naive when a period is known."""
    models = [Naive(), Mean(), Drift()]
    if period:
        models.append(SeasonalNaive(period=period))
    return models


def print_header(path, series, horizon, step, folds, first_cut):
    """Print what was read and how it was split."""
    span = f"{series.timestamps[0].date()} to {series.timestamps[-1].date()}"
    print(f"{path}: {len(series.values)} points, {describe_freq(series.freq)}, {span}")
    print(f"horizon {horizon}, step {step}, {folds} folds, first cut at {first_cut}")
    print()


def print_table(summary):
    """Print one row per model, best MASE first."""
    head = f"{'model':<20}{'MAE':>9}{'RMSE':>9}{'MASE':>9}{'median':>9}{'won':>7}"
    print(head)
    print("-" * len(head))

    for name, row in sorted(summary.items(), key=lambda item: item[1]["mase"]):
        print(
            f"{name:<20}{row['mae']:>9.1f}{row['rmse']:>9.1f}"
            f"{row['mase']:>9.4f}{row['mase_median']:>9.4f}"
            f"{row['beat_baseline']:>6.0%}"
        )

    print()
    print("MASE below 1 beats the naive baseline. Naive scores 1.0000 by construction.")
    print("'won' is the share of folds where the model beat naive, which the mean hides.")

    dropped = max(row["mase_dropped"] for row in summary.values())
    if dropped:
        print(f"{dropped} folds had a flat test block, so MASE was undefined there.")


def write_plot(path, series, horizon):
    """Plot the series with the last test block marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(11, 4))
    axes.plot(series.timestamps, series.values, linewidth=0.9, label="actual")
    axes.axvspan(
        series.timestamps[-horizon], series.timestamps[-1],
        alpha=0.15, label=f"last test block ({horizon} steps)",
    )
    axes.set_title("Series and the final test block")
    axes.legend(loc="upper left")
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    print(f"wrote {path}")


def run(args):
    """Load, evaluate, print, and optionally plot."""
    series = load_series(args.path, args.time, args.value, fill_gaps=args.fill_gaps)

    period = args.period if args.period else default_period(series.freq)
    models = build_models(period)
    first_cut = max(model.min_train_size for model in models)

    results = evaluate(series.values, models, args.horizon, args.step)
    summary = aggregate(results)

    print_header(
        args.path, series, args.horizon, args.step or args.horizon,
        folds=next(iter(summary.values()))["folds"], first_cut=first_cut,
    )
    print_table(summary)

    if args.plot:
        write_plot(args.plot, series, args.horizon)


def build_parser():
    """Build the argument parser for `python -m tsbench`."""
    parser = argparse.ArgumentParser(
        prog="python -m tsbench",
        description="Score forecasting models against a naive baseline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="evaluate models on a CSV")
    run_parser.add_argument("path", help="path to the CSV")
    run_parser.add_argument("--time", required=True, help="timestamp column")
    run_parser.add_argument("--value", required=True, help="column to forecast")
    run_parser.add_argument("--horizon", type=int, default=7,
                            help="steps ahead per fold (default 7)")
    run_parser.add_argument("--step", type=int, default=None,
                            help="how far the cut advances (default: horizon)")
    run_parser.add_argument("--period", type=int, default=None,
                            help="seasonal cycle length (default: from the spacing)")
    run_parser.add_argument("--fill-gaps", action="store_true",
                            help="interpolate missing rows instead of refusing")
    run_parser.add_argument("--plot", default=None, help="write a PNG here")
    return parser


def main(argv=None):
    """Entry point: parse arguments and run, turning refusals into exits."""
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except ValueError as error:
        raise SystemExit(f"error: {error}")
