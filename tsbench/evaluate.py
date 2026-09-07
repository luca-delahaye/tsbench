"""The loop: run models over folds and score them.

The only file that slices the series. Everything else receives arrays and
never sees where they came from, which is why the "a model cannot see the
future" guarantee is reviewable in two lines rather than spread out.
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
    """Score every model on the same folds. Returns per-fold scores by name.

        {"Naive": {"mae": [...], "rmse": [...], "mase": [...]}, ...}

    One entry per fold in each list, oldest fold first. Reducing those lists
    to single numbers is aggregate()'s job, not this one's -- the spread
    across folds is the interesting part and must survive to be looked at.

    Every model sees identical folds. The first cut point is the largest
    min_train_size in the run, so adding SeasonalNaive(24) pushes everyone
    forward together. Without that, MASE would be a ratio of two errors
    measured on different data, which is not a comparison at all.

    A fold whose test block is flat scores MASE as None: the baseline was
    exactly right, so the ratio would divide by zero. MAE and RMSE are still
    recorded for that fold.
    """
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
        # The only two cuts made anywhere in the project. train stops at
        # train_end, so no model can see the block it is scored on.
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
    """Reduce per-fold scores to one row per model.

        {"Naive": {"folds": 103, "mae": 921.3, "rmse": 1104.2,
                   "mase": 1.0, "mase_median": 1.0,
                   "beat_baseline": 0.0, "mase_dropped": 0}, ...}

    Both the mean and the median MASE are reported, and so is the share of
    folds the model actually won. A model can lose on the mean while winning
    half the folds -- a few bad folds drag an average that a median and a win
    rate do not hide. One number per model would be the easiest thing in the
    world to mislead with.
    """
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
