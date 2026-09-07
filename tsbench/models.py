"""Forecasting models, all behind one interface.

    model.fit(y_train)       here is the past, work out your recipe
    model.predict(horizon)   give me the next N numbers

evaluate.py loops over a list of these and never asks what kind it is
holding, so a baseline is scored by exactly the same code path as a real
model. That is what makes the comparison honest rather than something to
be trusted.

Positions only, no dates. Seasonality arrives as an integer period.
"""

from abc import ABC, abstractmethod

import numpy as np


class Model(ABC):
    """What every model shares, so each one is written as only its recipe.

    Subclasses implement _fit and _predict. The public fit and predict wrap
    them with the checks every model would otherwise repeat, so a model added
    later cannot forget them.
    """

    _fitted = False

    @property
    def min_train_size(self) -> int:
        """Fewest training points this model can say anything from.

        The model only declares the constraint. evaluate.py takes the largest
        across every model in a run and uses it as the floor for all of them,
        so the folds stay identical -- which is what MASE needs to mean
        anything.
        """
        return 1

    def fit(self, y_train: np.ndarray) -> "Model":
        """Learn from the training window. Returns self, so calls can chain."""
        y_train = np.asarray(y_train, dtype=float)
        if len(y_train) < self.min_train_size:
            raise ValueError(
                f"{type(self).__name__} needs at least {self.min_train_size} "
                f"training points, got {len(y_train)}"
            )
        self._fit(y_train)
        self._fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        """Forecast the next `horizon` points. Always returns an array."""
        if horizon < 1:
            raise ValueError(f"horizon must be at least 1, got {horizon}")
        if not self._fitted:
            raise ValueError(
                f"{type(self).__name__} must be fitted before predicting"
            )
        return self._predict(horizon)

    @abstractmethod
    def _fit(self, y_train: np.ndarray) -> None: ...

    @abstractmethod
    def _predict(self, horizon: int) -> np.ndarray: ...


class Naive(Model):
    """Tomorrow = today. The bar, and surprisingly hard to beat."""

    def _fit(self, y_train):
        self._last = y_train[-1]

    def _predict(self, horizon):
        return np.full(horizon, self._last)


class Mean(Model):
    """Tomorrow = the average of everything so far."""

    def _fit(self, y_train):
        self._mean = np.mean(y_train)

    def _predict(self, horizon):
        return np.full(horizon, self._mean)


class SeasonalNaive(Model):
    """Tomorrow = the same point one period ago, cycling if asked for more.

    period is an integer offset, never a calendar: 7 for a weekly cycle in
    daily data, 24 for a daily cycle in hourly data, 168 for a weekly cycle
    in hourly data.
    """

    def __init__(self, period):
        if period < 1:
            raise ValueError(f"period must be at least 1, got {period}")
        self.period = period

    @property
    def min_train_size(self) -> int:
        return self.period

    def _fit(self, y_train):
        self._season = y_train[-self.period:]

    def _predict(self, horizon):
        return self._season[np.arange(horizon) % self.period]


class Drift(Model):
    """Tomorrow = today plus the average step taken so far.

    The slope (last - first) / (n - 1) is exactly the mean one-step change:
    the interior terms telescope away. So it is blind to shape -- a series
    that rose and crashed back gets the same forecast as one that never
    moved. That is fine for a baseline, and worth knowing when reading one.
    """

    @property
    def min_train_size(self) -> int:
        return 2

    def _fit(self, y_train):
        self._slope = (y_train[-1] - y_train[0]) / (len(y_train) - 1)
        self._last = y_train[-1]

    def _predict(self, horizon):
        return (np.arange(1, horizon + 1) * self._slope) + self._last
