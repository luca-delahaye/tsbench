"""The loop: run models over folds and score them.

The only file that slices the series, so the guarantee that a model never
sees the block it is scored on lives in two reviewable lines.
"""

import numpy as np

from tsbench.metrics import mae, mase, rmse
from tsbench.models import Naive
from tsbench.splits import make_folds


def run_folds(values, model, horizon, min_train_size, step=None):
    """Return one MAE per fold, oldest fold first."""

    folds = make_folds(len(values), horizon, step, min_train_size)
    scores = []

    for fold in folds:
        train = values[fold.train_start:fold.train_end]
        actual = values[fold.test_start:fold.test_end]

        model.fit(train)
        predicted = model.predict(horizon)
        scores.append(mae(actual, predicted))

    return scores


def evaluate(values, models, horizon, step=None):
    """Score every model on identical folds, returning per-fold scores by name."""
    values = np.asarray(values, dtype=float)
    if not models:
        raise ValueError("need at least one model to evaluate")

    names = [model.name for model in models]
    if len(set(names)) != len(names):
        raise ValueError(f"two models share a name: {names}")

    baseline = Naive()
    min_train_size = max(model.min_train_size for model in [baseline, *models])
    folds = make_folds(len(values), horizon, step, min_train_size)

    results = {name: {"mae": [], "rmse": [], "mase": []} for name in names}

    for fold in folds:
        train = values[fold.train_start:fold.train_end]
        actual = values[fold.test_start:fold.test_end]

        baseline_predicted = baseline.fit(train).predict(horizon)
        baseline_was_exact = mae(actual, baseline_predicted) == 0

        for model in models:
            predicted = model.fit(train).predict(horizon)
            scores = results[model.name]
            scores["mae"].append(mae(actual, predicted))
            scores["rmse"].append(rmse(actual, predicted))
            scores["mase"].append(
                None if baseline_was_exact
                else mase(actual, predicted, baseline_predicted)
            )

    return results


def aggregate(results):
    """Reduce per-fold scores to one summary row per model."""
    summary = {}
    for name, scores in results.items():
        scored = [value for value in scores["mase"] if value is not None]
        summary[name] = {
            "folds": len(scores["mae"]),
            "mae": float(np.mean(scores["mae"])),
            "rmse": float(np.mean(scores["rmse"])),
            "mase": float(np.mean(scored)) if scored else None,
            "mase_median": float(np.median(scored)) if scored else None,
            "beat_baseline": float(np.mean([v < 1 for v in scored])) if scored else None,
            "mase_dropped": len(scores["mase"]) - len(scored),
        }
    return summary
