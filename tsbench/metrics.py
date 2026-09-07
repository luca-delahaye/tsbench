"""Scoring: how wrong a forecast was, as a single number.

Arrays in, one float out: no models, no folds. Combining fold scores is
evaluate.py's job. Argument order is always (actual, predicted).
"""

import numpy as np


def _check_pair(
        actual: np.ndarray,
        predicted: np.ndarray,
        name: str = "predicted",
) -> None:
    """Refuse arrays that cannot be compared: wrong length, or empty."""
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
    """Like MAE, but one big miss costs more than several small ones."""
    _check_pair(actual, predicted)
    return np.sqrt(np.mean((actual - predicted) ** 2))


def mase(
        actual: np.ndarray,
        predicted: np.ndarray,
        baseline_predicted: np.ndarray,
) -> float:
    """This forecast's MAE divided by the baseline's, on the same test block."""
    _check_pair(actual, predicted)
    _check_pair(actual, baseline_predicted, name="baseline_predicted")

    baseline_error = mae(actual, baseline_predicted)
    if baseline_error == 0:
        raise ValueError(
            "the baseline forecast was exactly right, so its error is zero and "
            "MASE would divide by it; this happens when the test block is flat"
        )
    return mae(actual, predicted) / baseline_error
