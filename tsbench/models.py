"""Forecasting models, all behind one interface.

    model.fit(y_train)       here is the past, work out your recipe
    model.predict(horizon)   give me the next N numbers

evaluate.py loops over a list of these and never asks what kind it is
holding, so a baseline is scored by exactly the same code path as a real
model. That is what makes the comparison honest rather than something to
be trusted.

Positions only, no dates. Seasonality arrives as an integer period.
"""

import numpy as np


class Naive:
    """Tomorrow = today. The bar, and surprisingly hard to beat."""

    def fit(self, y_train):
        self._last = y_train[-1]
        return self

    def predict(self, horizon):
        return np.full(horizon, self._last)


class Mean:
    """Tomorrow = the average of everything so far."""

    def fit(self, y_train):
        self._mean = np.mean(y_train)
        return self

    def predict(self, horizon):
        return np.full(horizon, self._mean)


class SeasonalNaive:
    """Tomorrow = the same point one period ago, cycling if asked for more.

    period is an integer offset, never a calendar: 7 for a weekly cycle in
    daily data, 24 for a daily cycle in hourly data, 168 for a weekly cycle
    in hourly data.
    """

    def __init__(self, period):
        if period < 1:
            raise ValueError(f"period must be at least 1, got {period}")
        self.period = period

    def fit(self, y_train):
        if len(y_train) < self.period:
            raise ValueError(
                f"SeasonalNaive needs at least {self.period} training points, "
                f"got {len(y_train)}"
            )
        self._season = y_train[-self.period:]
        return self

    def predict(self, horizon):
        return self._season[np.arange(horizon) % self.period]


class Drift:
    """Tomorrow = today plus the average step taken so far.

    The slope (last - first) / (n - 1) is exactly the mean one-step change:
    the interior terms telescope away. So it is blind to shape -- a series
    that rose and crashed back gets the same forecast as one that never
    moved. That is fine for a baseline, and worth knowing when reading one.
    """

    def fit(self, y_train):
        if len(y_train) < 2:
            raise ValueError(
                f"Drift needs at least 2 training points, "
                f"got {len(y_train)}"
            )
        self._slope = (y_train[-1] - y_train[0])/(len(y_train) - 1)
        self._last = y_train[-1]
        return self

    def predict(self, horizon):
        return (np.arange(1, horizon + 1) * self._slope) + self._last
    