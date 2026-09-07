"""Tests for the forecasting models.

Expected forecasts are worked out by hand from one training window throughout,
so the four models can be read side by side.
"""

import numpy as np
import pytest

from tsbench.models import Drift, Mean, Model, Naive, SeasonalNaive

# Seven points rising by 10. Last value 160, mean 910 / 7 = 130,
# slope (160 - 100) / 6 = 10.
TRAIN = np.array([100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0])


def test_naive_repeats_the_last_value():
    assert np.array_equal(Naive().fit(TRAIN).predict(3), np.full(3, 160.0))


def test_mean_repeats_the_average():
    """910 / 7 = 130"""
    assert np.allclose(Mean().fit(TRAIN).predict(3), np.full(3, 130.0))


def test_drift_continues_the_trend():
    """Slope 10 per step from 160: 170, 180, 190."""
    assert np.allclose(Drift().fit(TRAIN).predict(3), np.array([170.0, 180.0, 190.0]))


def test_drift_slope_uses_the_gaps_not_the_points():
    """Seven points have six gaps: 60 / 6 = 10, not 60 / 7 = 8.57."""
    assert np.allclose(Drift().fit(TRAIN).predict(1), np.array([170.0]))


def test_drift_goes_downhill_too():
    """Slope (60 - 100) / 2 = -20 from 60."""
    falling = np.array([100.0, 80.0, 60.0])
    assert np.allclose(Drift().fit(falling).predict(2), np.array([40.0, 20.0]))


def test_drift_is_blind_to_shape():
    """A series ending where it started forecasts flat, whatever happened between.

    Not a bug: the slope is the mean one-step change, and those changes sum to
    zero here. Worth asserting so the limitation is documented, not discovered.
    """
    round_trip = np.array([100.0, 500.0, 20.0, 300.0, 100.0])
    assert np.allclose(Drift().fit(round_trip).predict(3), np.full(3, 100.0))


def test_seasonal_naive_replays_the_last_period():
    """Period 7 with 7 training points: the forecast is the window itself."""
    forecast = SeasonalNaive(period=7).fit(TRAIN).predict(3)
    assert np.array_equal(forecast, np.array([100.0, 110.0, 120.0]))


def test_seasonal_naive_steps_forward_within_the_period():
    """Each step looks one period back from itself: 24, 25, 26 -- not 24 thrice."""
    forecast = SeasonalNaive(period=24).fit(np.arange(48.0)).predict(3)
    assert np.array_equal(forecast, np.array([24.0, 25.0, 26.0]))


def test_seasonal_naive_cycles_past_one_period():
    """Last 3 of [1..6] is [4, 5, 6], replayed: 4 5 6 4 5 6 4."""
    train = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    forecast = SeasonalNaive(period=3).fit(train).predict(7)
    assert np.array_equal(forecast, np.array([4.0, 5.0, 6.0, 4.0, 5.0, 6.0, 4.0]))


def test_flat_series_is_predicted_flat_by_every_model():
    """Nothing is happening, so nothing should be forecast to happen."""
    flat = np.full(10, 42.0)
    for model in (Naive(), Mean(), Drift(), SeasonalNaive(period=4)):
        assert np.allclose(model.fit(flat).predict(5), 42.0)


def test_every_model_answers_the_same_two_calls():
    """The interface is the product: evaluate.py must not special-case anyone."""
    for model in (Naive(), Mean(), Drift(), SeasonalNaive(period=3)):
        forecast = model.fit(TRAIN).predict(5)
        assert isinstance(forecast, np.ndarray)
        assert len(forecast) == 5


def test_every_model_is_built_without_data():
    """evaluate.py builds the list before any CSV is read."""
    for model in (Naive(), Mean(), Drift(), SeasonalNaive(period=3)):
        assert hasattr(model, "fit")


def test_too_little_history_is_refused():
    with pytest.raises(ValueError, match="needs at least 2 training points"):
        Drift().fit(np.array([1.0]))

    with pytest.raises(ValueError, match="needs at least 24 training points"):
        SeasonalNaive(period=24).fit(np.arange(10.0))


def test_nonsense_period_is_refused():
    with pytest.raises(ValueError, match="period must be at least 1"):
        SeasonalNaive(period=0)


def test_models_declare_what_they_need():
    """Each states its own floor; evaluate.py takes the largest across a run."""
    assert Naive().min_train_size == 1
    assert Mean().min_train_size == 1
    assert Drift().min_train_size == 2
    assert SeasonalNaive(period=24).min_train_size == 24


def test_predicting_before_fitting_is_refused():
    """Was an AttributeError naming a private field. Now it says what is wrong."""
    with pytest.raises(ValueError, match="must be fitted before predicting"):
        Naive().predict(3)


def test_nonsense_horizon_is_refused():
    """Horizon 0 used to return an empty array with no complaint at all."""
    for horizon in (0, -5):
        with pytest.raises(ValueError, match="horizon must be at least 1"):
            Naive().fit(TRAIN).predict(horizon)


def test_the_interface_itself_cannot_be_instantiated():
    """A model that forgets _predict fails here, not deep inside the fold loop."""
    with pytest.raises(TypeError):
        Model()

    class Forgetful(Model):
        def _fit(self, y_train):
            pass

    with pytest.raises(TypeError):
        Forgetful()
