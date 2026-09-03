"""Rolling-origin splitting: which positions to train on, which to test on.

Pure arithmetic on the length of the series. No data, no dates, no numpy.

Every range is half-open, [start, end), so two adjacent ranges share a number
and can neither overlap nor leave a gap.
"""

from typing import NamedTuple


class Fold(NamedTuple):
    """One experiment: train on [train_start, train_end), test on [test_start, test_end)."""

    train_start: int
    train_end: int
    test_start: int
    test_end: int


def _check_parameters(n: int, horizon: int, step: int, min_train_size: int) -> None:
    """Refuse arguments whose arithmetic is legal but meaningless.

    A step of 0 never advances the cut point; a horizon of 0 tests nothing.
    Neither raises on its own -- both return something that looks like a
    result, which is worse than a crash.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon}")
    if step < 1:
        raise ValueError(f"step must be at least 1, got {step}")
    if min_train_size < 1:
        raise ValueError(f"min_train_size must be at least 1, got {min_train_size}")
    if n < min_train_size + horizon:
        raise ValueError(
            f"series too short: {n} points, but one fold needs {min_train_size} "
            f"to train on and {horizon} to test on, so {min_train_size + horizon} "
            f"at minimum"
        )


def make_folds(
        n: int,
        horizon: int,
        step: int | None = None,
        min_train_size: int = 1,
) -> list[Fold]:
    """Return the folds for a series of n points, oldest first.

    The training window expands: every fold trains on everything from position
    0 up to its own cut point. step defaults to horizon, which makes the test
    blocks tile exactly -- nothing tested twice, nothing skipped.

    Incomplete folds are never emitted, so trailing positions may go untested.
    A short final fold would be scored over fewer points than the others, and
    averaging that with them is arithmetic nonsense.
    """
    if step is None:
        step = horizon
    _check_parameters(n, horizon, step, min_train_size)

    folds = []
    cut = min_train_size
    while cut + horizon <= n:
        folds.append(Fold(0, cut, cut, cut + horizon))
        cut += step
    return folds
