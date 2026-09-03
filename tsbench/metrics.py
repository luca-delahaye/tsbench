"""Scoring: how wrong a forecast was, as a single number.

Arrays in, one float out. No models, no folds -- combining fold scores is
evaluate.py's job, because that is where the folds exist.

Argument order is always (actual, predicted).
"""

import numpy as np


def _check_pair(
        actual: np.ndarray,
        predicted: np.ndarray,
        name: str = "predicted",
) -> None:
    """Refuse arrays that cannot be compared.

    Empty ones are the dangerous case: they score to nan, and a nan travels
    silently through a ratio into the results table.
    """
    if len(actual) != len(predicted):
        raise ValueError(
            f"actual has {len(actual)} points, {name} has {len(predicted)}"
        )
    if len(actual) == 0:
        raise ValueError("cannot score an empty forecast")


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Average size of a miss, in the series' units."""
    _check_pair(actual, predicted)
    return np.abs(actual - predicted).mean()


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Like MAE, but one big miss costs more than several small ones adding to
    the same total.

    Read the two together: RMSE near MAE means steadily mediocre, RMSE far
    above it means usually fine with rare disasters. Different problems.
    """
    _check_pair(actual, predicted)
    return np.sqrt(np.mean((actual - predicted) ** 2))


def mase(
        actual: np.ndarray,
        predicted: np.ndarray,
        baseline_predicted: np.ndarray,
) -> float:
    """This forecast's MAE divided by the baseline's, on the same test block.

    Below 1 beat the baseline, above 1 lost to it. The units cancel, so unlike
    MAE this number means something on its own.

    Both sides predict the same block over the same horizon, so it is a fair
    fight. Scoring the baseline against itself gives exactly 1.0 on any data,
    which checks the harness for free.

    This is the relative form, not Hyndman's MASE -- his scales by the one-step
    naive error measured on the training data. Different number, so the README
    has to say which one it reports.

    Raises if the baseline was perfect. That means a flat test block: a stuck
    sensor, a closed market. The caller decides what to do with such a fold.
    """
    _check_pair(actual, predicted)
    _check_pair(actual, baseline_predicted, name="baseline_predicted")

    baseline_error = mae(actual, baseline_predicted)
    if baseline_error == 0:
        raise ValueError(
            "the baseline forecast was exactly right, so its error is zero and "
            "MASE would divide by it; this happens when the test block is flat"
        )
    return mae(actual, predicted) / baseline_error
