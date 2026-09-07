"""Tests for the fold loop.

The linear series 0, 1, 2, ... 99 is used throughout because every model's
error on it can be worked out by hand. Naive predicts the last value for all
7 steps, so its errors are 1, 2, 3, 4, 5, 6, 7 and its MAE is 4 in every fold.
Drift extrapolates the line exactly, so it scores 0.
"""

import numpy as np
import pytest

from tsbench.evaluate import evaluate, run_folds
from tsbench.models import Drift, Mean, Naive, SeasonalNaive

LINEAR = np.arange(100.0)


def test_run_folds_known_answer():
    """Naive on a line: errors 1..7 every fold, so MAE is (1+..+7)/7 = 4."""
    scores = run_folds(LINEAR, Naive(), horizon=7, min_train_size=30)
    assert scores == [4.0] * len(scores)
    assert len(scores) == 10


def test_drift_is_exact_on_a_straight_line():
    """The strongest end-to-end check: slicing, fitting and scoring all agree."""
    scores = run_folds(LINEAR, Drift(), horizon=7, min_train_size=30)
    assert scores == [0.0] * len(scores)


def test_the_baseline_scores_exactly_one():
    """Naive is what MASE divides by, so it must score 1.0 in every fold.

    Holds on any series. A number other than 1.0 here means the harness fed
    the baseline different data from the model, which would make every other
    MASE meaningless.
    """
    results = evaluate(LINEAR, [Naive()], horizon=7)
    assert results["Naive"]["mase"] == [1.0] * len(results["Naive"]["mase"])


def test_every_model_is_scored_on_the_same_folds():
    models = [Naive(), Mean(), Drift(), SeasonalNaive(7)]
    results = evaluate(LINEAR, models, horizon=7)

    counts = {len(results[m.name]["mae"]) for m in models}
    assert len(counts) == 1, "models were scored on different numbers of folds"


def test_the_hungriest_model_sets_the_floor_for_everyone():
    """Adding SeasonalNaive(24) pushes the first cut point forward for all."""
    alone = evaluate(LINEAR, [Naive()], horizon=7)
    together = evaluate(LINEAR, [Naive(), SeasonalNaive(24)], horizon=7)

    assert len(together["Naive"]["mae"]) < len(alone["Naive"]["mae"])
    assert len(together["Naive"]["mae"]) == len(together["SeasonalNaive(24)"]["mae"])


def test_results_carry_one_entry_per_fold_per_metric():
    results = evaluate(LINEAR, [Naive(), Drift()], horizon=7)
    for scores in results.values():
        assert set(scores) == {"mae", "rmse", "mase"}
        assert len(scores["mae"]) == len(scores["rmse"]) == len(scores["mase"])


def test_a_flat_series_scores_mase_as_none():
    """The baseline is exactly right, so the ratio would divide by zero.

    MAE and RMSE are still recorded: only the ratio is undefined.
    """
    results = evaluate(np.full(100, 42.0), [Naive(), Mean()], horizon=7)
    for scores in results.values():
        assert all(m is None for m in scores["mase"])
        assert all(e == 0.0 for e in scores["mae"])


def test_no_models_is_refused():
    with pytest.raises(ValueError, match="need at least one model"):
        evaluate(LINEAR, [], horizon=7)


def test_two_models_sharing_a_name_are_refused():
    """Otherwise one silently overwrites the other's row in the results."""
    with pytest.raises(ValueError, match="share a name"):
        evaluate(LINEAR, [Naive(), Naive()], horizon=7)


def test_two_seasonal_models_are_told_apart_by_period():
    results = evaluate(LINEAR, [SeasonalNaive(7), SeasonalNaive(14)], horizon=7)
    assert set(results) == {"SeasonalNaive(7)", "SeasonalNaive(14)"}


# A series with real structure: trend, a weekly cycle, and noise. Fixed seed,
# so the numbers below are reproducible.
def _synthetic(n=200):
    rng = np.random.default_rng(0)
    t = np.arange(n)
    return 100 + 0.5 * t + 20 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 5, n)


MODELS = lambda: [Naive(), Mean(), Drift(), SeasonalNaive(7)]


def test_corrupting_the_future_changes_nothing():
    """The lookahead test. The single most valuable test in this repo.

    Score the first 60 points. Then score a 200-point series whose first 60
    points are identical and whose remaining 140 are garbage. Every fold that
    lives entirely inside the first 60 must produce byte-identical scores.

    Not similar -- identical. Those folds were never supposed to see the
    corrupted region, so if one digit moves, something read the future.

    This catches all three leaks at once, without caring how careful anyone
    was being: a centred rolling window, an off-by-one in a trailing window,
    and a scaler fitted on the whole series before splitting.
    """
    values = _synthetic(200)
    cut = 60

    corrupted = values.copy()
    corrupted[cut:] = 1e9

    clean = evaluate(values[:cut], MODELS(), horizon=7)
    dirty = evaluate(corrupted, MODELS(), horizon=7)

    for name, scores in clean.items():
        folds = len(scores["mae"])
        assert folds > 0, "the test is vacuous if no fold fits in the clean part"
        for metric in ("mae", "rmse", "mase"):
            assert dirty[name][metric][:folds] == scores[metric], (
                f"{name} {metric} changed when only the future changed"
            )


def test_the_lookahead_test_has_teeth():
    """A test that cannot fail proves nothing.

    Corrupting a point inside every training window must change the scores.
    If this passes while the test above also passes, the test above is
    measuring something real rather than comparing two identical runs.
    """
    values = _synthetic(200)

    corrupted = values.copy()
    corrupted[10] = 1e9

    clean = evaluate(values, MODELS(), horizon=7)
    dirty = evaluate(corrupted, MODELS(), horizon=7)

    assert dirty["Mean"]["mae"] != clean["Mean"]["mae"]
