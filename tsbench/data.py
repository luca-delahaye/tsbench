"""Reading a CSV into a series the rest of the project can trust.

An airlock, not a converter: it refuses anything that would break "position
N is N steps in time". The only file that touches pandas or knows a date.
"""

from typing import NamedTuple

import numpy as np
import pandas as pd


class Series(NamedTuple):
    """A validated series: the numbers, when they happened, and the spacing."""

    values: np.ndarray
    timestamps: pd.DatetimeIndex
    freq: pd.Timedelta


def _read_columns(path, time_column, value_column):
    """Read the two columns we care about, keeping their text as written."""
    frame = pd.read_csv(path, keep_default_na=False)
    for column in (time_column, value_column):
        if column not in frame.columns:
            raise ValueError(
                f"no column named {column!r} in {path}; "
                f"found {list(frame.columns)}"
            )
    return frame[[time_column, value_column]]


def _parse_timestamps(frame, time_column):
    """Turn the time column into real timestamps, naming the first bad row."""
    stamps = pd.to_datetime(frame[time_column], errors="coerce")
    if stamps.isna().any():
        row = int(stamps.isna().idxmax())
        raise ValueError(
            f"row {row}: {frame[time_column].iloc[row]!r} is not a timestamp"
        )
    return stamps


def _parse_values(frame, value_column, stamps):
    """Turn the value column into floats, naming the first bad row."""
    values = pd.to_numeric(frame[value_column], errors="coerce")
    if values.isna().any():
        row = int(values.isna().idxmax())
        raise ValueError(
            f"row {row} ({stamps.iloc[row]}): "
            f"{frame[value_column].iloc[row]!r} is not a number"
        )
    return values


def _check_ordered(stamps):
    """Refuse unsorted or duplicated timestamps, naming the offending row."""
    if stamps.duplicated().any():
        row = int(stamps.duplicated().idxmax())
        raise ValueError(f"row {row}: {stamps.iloc[row]} appears twice")
    if not stamps.is_monotonic_increasing:
        row = int((stamps.diff() < pd.Timedelta(0)).idxmax())
        raise ValueError(
            f"row {row}: {stamps.iloc[row]} comes before the row above it"
        )


def _spacing(stamps):
    """The gap between consecutive rows when nothing is missing."""
    if len(stamps) < 2:
        raise ValueError("need at least two rows to know the spacing")
    return stamps.diff().dropna().min()


def _check_no_gaps(stamps, freq):
    """Refuse a series with a hole in it, naming where and how big."""
    steps = stamps.diff()
    gaps = steps > freq
    if gaps.any():
        row = int(gaps.idxmax())
        missing = int(steps.iloc[row] / freq) - 1
        raise ValueError(
            f"row {row}: {missing} missing at {stamps.iloc[row - 1]} "
            f"(jumps to {stamps.iloc[row]}); "
            f"{int(gaps.sum())} gaps in total. Pass --fill-gaps to interpolate them."
        )


def _fill_gaps(values, stamps, freq):
    """Put the missing rows back, interpolating between the neighbours."""
    complete = pd.date_range(stamps.iloc[0], stamps.iloc[-1], freq=freq)
    filled = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(stamps))
    filled = filled.reindex(complete).interpolate()
    added = len(complete) - len(stamps)
    print(f"filled {added} missing rows by interpolation")
    return filled.to_numpy(), complete


def load_series(path, time_column, value_column, fill_gaps=False) -> Series:
    """Read a CSV into a validated Series, refusing anything untrustworthy."""
    frame = _read_columns(path, time_column, value_column)
    stamps = _parse_timestamps(frame, time_column)
    values = _parse_values(frame, value_column, stamps)

    _check_ordered(stamps)
    freq = _spacing(stamps)

    if fill_gaps:
        numbers, index = _fill_gaps(values, stamps, freq)
    else:
        _check_no_gaps(stamps, freq)
        numbers, index = values.to_numpy(dtype=float), pd.DatetimeIndex(stamps)

    return Series(values=numbers.astype(float), timestamps=index, freq=freq)
