"""Tests for the rolling-origin splitter.

Expected folds are written out by hand, so the test cannot agree with a
wrong implementation by repeating its arithmetic.
"""

import pytest

from tsbench.splits import Fold, make_folds


def test_known_answer():
    """23 points, horizon 4, at least 5 to train on."""
    assert make_folds(23, 4, min_train_size=5) == [
        Fold(0, 5, 5, 9),
        Fold(0, 9, 9, 13),
        Fold(0, 13, 13, 17),
        Fold(0, 17, 17, 21),
    ]


def test_step_defaults_to_horizon():
    assert make_folds(23, 4, min_train_size=5) == make_folds(
        23, 4, step=4, min_train_size=5
    )


def test_test_blocks_tile_exactly():
    """With the default step, each test block starts where the last one ended."""
    folds = make_folds(100, 7, min_train_size=30)
    for earlier, later in zip(folds, folds[1:]):
        assert later.test_start == earlier.test_end


def test_training_stops_where_testing_starts():
    """The cut point is one number, so the model can never see what it predicts."""
    for fold in make_folds(100, 7, step=3, min_train_size=30):
        assert fold.train_end == fold.test_start


def test_every_fold_is_complete():
    """No short final fold: scores must be computed over equal numbers of points."""
    n, horizon = 100, 7
    for fold in make_folds(n, horizon, min_train_size=30):
        assert fold.test_end - fold.test_start == horizon
        assert fold.test_end <= n


def test_training_window_expands():
    folds = make_folds(100, 7, min_train_size=30)
    for earlier, later in zip(folds, folds[1:]):
        assert later.train_start == 0
        assert later.train_end > earlier.train_end


def test_shortest_series_that_still_works():
    """Exactly min_train_size + horizon points must yield exactly one fold."""
    assert make_folds(9, 4, min_train_size=5) == [Fold(0, 5, 5, 9)]


def test_series_too_short_is_refused():
    with pytest.raises(ValueError, match="series too short"):
        make_folds(8, 4, min_train_size=5)


def test_nonsense_parameters_are_refused():
    """Each of these would otherwise return something that looks like a result."""
    with pytest.raises(ValueError):
        make_folds(100, 0, min_train_size=30)

    with pytest.raises(ValueError):
        make_folds(100, 7, step=0, min_train_size=30)

    with pytest.raises(ValueError):
        make_folds(100, 7, min_train_size=0)
