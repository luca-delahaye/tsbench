"""Forecasting models, all behind one interface.

fit(y_train) then predict(horizon). evaluate.py loops over these without
asking what kind each one is, so a baseline is scored like any other model.
"""

from abc import ABC, abstractmethod

import numpy as np


class Model(ABC):
    """What every model shares, so each one is written as only its recipe."""

    _fitted = False

    @property
    def name(self) -> str:
        """Label for the results table; two models in a run need two names."""
        return type(self).__name__

    @property
    def min_train_size(self) -> int:
        """Fewest training points this model can say anything from."""
        return 1

    def fit(self, y_train: np.ndarray) -> "Model":
        """Learn from the training window and return self, so calls chain."""
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
        """Forecast the next `horizon` points, always as an array."""
        if horizon < 1:
            raise ValueError(f"horizon must be at least 1, got {horizon}")
        if not self._fitted:
            raise ValueError(
                f"{type(self).__name__} must be fitted before predicting"
            )
        return self._predict(horizon)

    @abstractmethod
    def _fit(self, y_train: np.ndarray) -> None:
        """Work out this model's recipe from the training window."""

    @abstractmethod
    def _predict(self, horizon: int) -> np.ndarray:
        """Apply the recipe to produce `horizon` values."""


class Naive(Model):
    """Tomorrow = today: the bar, and surprisingly hard to beat."""

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
    """Tomorrow = the same point one period ago, cycling for longer horizons."""

    def __init__(self, period):
        if period < 1:
            raise ValueError(f"period must be at least 1, got {period}")
        self.period = period

    @property
    def name(self) -> str:
        return f"SeasonalNaive({self.period})"

    @property
    def min_train_size(self) -> int:
        return self.period

    def _fit(self, y_train):
        self._season = y_train[-self.period:]

    def _predict(self, horizon):
        return self._season[np.arange(horizon) % self.period]


class Drift(Model):
    """Tomorrow = today plus the mean one-step change, so blind to shape."""

    @property
    def min_train_size(self) -> int:
        return 2

    def _fit(self, y_train):
        self._slope = (y_train[-1] - y_train[0]) / (len(y_train) - 1)
        self._last = y_train[-1]

    def _predict(self, horizon):
        return (np.arange(1, horizon + 1) * self._slope) + self._last
