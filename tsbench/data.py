"""Reading a CSV into a series the rest of the project can trust.

Not a converter, an airlock. Every other file is built on one assumption:

    position N is exactly N steps in time.

That is a lie unless somebody checks it. If three hours are missing from an
hourly file, "look back 24 positions" quietly lands 27 hours ago, forever,
and nothing tells you. So the checks happen once, here, at the door.

This is the only file that touches pandas, and the only one that knows what
a date is. Everything downstream counts positions.
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
    """Read the two columns we care about, or say which one is missing.

    keep_default_na=False stops pandas quietly turning 'n/a', 'NULL' and
    friends into NaN while reading. We want the original text to survive
    that far, so the error message can quote what was actually in the file.
    """
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
    """Turn the value column into floats, naming the first bad row.

    A single stray 'n/a' makes pandas read the whole column as text, so this
    has to be checked rather than assumed.
    """
    values = pd.to_numeric(frame[value_column], errors="coerce")
    if values.isna().any():
        row = int(values.isna().idxmax())
        raise ValueError(
            f"row {row} ({stamps.iloc[row]}): "
            f"{frame[value_column].iloc[row]!r} is not a number"
        )
    return values


def _check_ordered(stamps):
    """Refuse unsorted or duplicated timestamps.

    Out of order means the past is no longer behind you in the array.
    Duplicates mean two rows for the same moment, so every lag after that
    point is off by one.
    """
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
    """Refuse a series with a hole in it, naming where and how big.

    Refusing is the default on purpose. Interpolating a gap and then testing
    over it scores the model against numbers we invented, and it will do
    beautifully. --fill-gaps exists, but it has to be typed.
    """
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
    """Read a CSV and hand back a series that the rest of the project can trust.

    Refuses anything that would break "position N is N steps in time":
    unparseable dates or numbers, duplicates, wrong order, or gaps. Errors
    name the row, because "gap at 2011-03-14 03:00" beats "gap at 1783".
    """
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
