"""Tests for the scoring functions.

Expected values are worked out by hand and shown in each docstring, since a
test that recomputes the implementation's own formula cannot catch it.
"""

import numpy as np
import pytest

from tsbench.metrics import mae, mase, rmse

ACTUAL = np.array([125.0, 128.0, 190.0, 122.0, 120.0])
PREDICTED = np.array([120.0, 130.0, 140.0, 125.0, 118.0])

NAIVE = np.array([118.0, 118.0, 118.0, 118.0, 118.0])


def test_mae_known_answer():
    """(5 + 2 + 50 + 3 + 2) / 5 = 62 / 5 = 12.4"""
    assert mae(ACTUAL, PREDICTED) == pytest.approx(12.4)


def test_rmse_known_answer():
    """sqrt((25 + 4 + 2500 + 9 + 4) / 5) = sqrt(2542 / 5) = sqrt(508.4)"""
    assert rmse(ACTUAL, PREDICTED) == pytest.approx(22.5477, abs=1e-4)


def test_rmse_far_exceeds_mae_when_one_miss_dominates():
    """The gap between them is the signal: a few disasters, not steady mediocrity."""
    assert rmse(ACTUAL, PREDICTED) > 1.5 * mae(ACTUAL, PREDICTED)


def test_rmse_equals_mae_when_every_miss_is_the_same_size():
    """Every error is exactly 3, so squaring and rooting changes nothing."""
    actual = np.array([10.0, 20.0, 30.0])
    predicted = np.array([13.0, 23.0, 33.0])
    assert mae(actual, predicted) == pytest.approx(3.0)
    assert rmse(actual, predicted) == pytest.approx(3.0)


def test_perfect_forecast_scores_zero():
    assert mae(ACTUAL, ACTUAL) == 0.0
    assert rmse(ACTUAL, ACTUAL) == 0.0


def test_mase_known_answer():
    """Baseline MAE 95 / 5 = 19, so MASE = 12.4 / 19 = 0.652632."""
    assert mase(ACTUAL, PREDICTED, NAIVE) == pytest.approx(0.652632, abs=1e-6)


def test_baseline_scored_against_itself_is_exactly_one():
    """An identity, not an approximation: mae(a, n) / mae(a, n)."""
    assert mase(ACTUAL, NAIVE, NAIVE) == 1.0

    noise = np.array([3.0, -40.0, 7.5, 0.25, 118.0])
    assert mase(ACTUAL, noise, noise) == 1.0


def test_mase_above_one_means_beaten_by_the_baseline():
    """Baseline misses by 10 each step, the model by 20. 20 / 10 = 2."""
    actual = np.array([10.0, 20.0])
    baseline = np.array([0.0, 10.0])
    predicted = np.array([30.0, 40.0])
    assert mase(actual, predicted, baseline) == pytest.approx(2.0)


def test_perfect_model_scores_zero():
    assert mase(ACTUAL, ACTUAL, NAIVE) == 0.0


def test_flat_test_block_is_refused():
    """The baseline is exactly right, so the ratio would divide by zero."""
    actual = np.array([10.0, 20.0])
    with pytest.raises(ValueError, match="baseline forecast was exactly right"):
        mase(actual, np.array([11.0, 21.0]), actual)


def test_mismatched_lengths_are_refused():
    """numpy would talk about broadcasting; this should talk about forecasts."""
    for metric in (mae, rmse):
        with pytest.raises(ValueError, match="actual has 5 points, predicted has 3"):
            metric(ACTUAL, PREDICTED[:3])


def test_a_wrong_length_baseline_names_the_baseline():
    """The message must point at the argument that was actually wrong."""
    with pytest.raises(ValueError, match="baseline_predicted has 3"):
        mase(ACTUAL, PREDICTED, NAIVE[:3])


def test_empty_forecast_is_refused():
    """Left alone this returns nan, which travels silently into the table."""
    empty = np.array([])
    for metric in (mae, rmse):
        with pytest.raises(ValueError, match="empty forecast"):
            metric(empty, empty)

    with pytest.raises(ValueError, match="empty forecast"):
        mase(empty, empty, empty)
