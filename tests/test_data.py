"""Tests for the CSV airlock.

Each test writes a tiny CSV to a temporary directory. The point of the file
is what it refuses, so most of these check a message rather than a value.
"""

import numpy as np
import pandas as pd
import pytest

from tsbench.data import load_series


def write_csv(directory, rows, header="date,value"):
    """Write rows to a CSV in a temporary directory and return its path."""
    path = directory / "series.csv"
    path.write_text(header + "\n" + "\n".join(rows) + "\n")
    return path


CLEAN = [
    "2024-01-01,10",
    "2024-01-02,20",
    "2024-01-03,30",
    "2024-01-04,40",
]


def test_reads_a_clean_file(tmp_path):
    series = load_series(write_csv(tmp_path, CLEAN), "date", "value")

    assert np.array_equal(series.values, np.array([10.0, 20.0, 30.0, 40.0]))
    assert series.freq == pd.Timedelta(days=1)
    assert len(series.timestamps) == 4


def test_spacing_is_read_from_the_data_not_assumed(tmp_path):
    """Hourly rows give an hourly freq without anyone saying so."""
    hourly = [
        "2024-01-01 00:00,10",
        "2024-01-01 01:00,20",
        "2024-01-01 02:00,30",
    ]
    series = load_series(write_csv(tmp_path, hourly), "date", "value")
    assert series.freq == pd.Timedelta(hours=1)


def test_a_gap_is_refused_and_located(tmp_path):
    """January 3rd is missing, so position arithmetic would stop meaning time."""
    gapped = ["2024-01-01,10", "2024-01-02,20", "2024-01-04,40"]

    with pytest.raises(ValueError, match="1 missing at 2024-01-02"):
        load_series(write_csv(tmp_path, gapped), "date", "value")


def test_gaps_can_be_filled_but_only_on_purpose(tmp_path):
    """Interpolating and then testing over the gap scores invented numbers."""
    gapped = ["2024-01-01,10", "2024-01-02,20", "2024-01-04,40"]
    series = load_series(write_csv(tmp_path, gapped), "date", "value", fill_gaps=True)

    assert len(series.values) == 4
    assert series.values[2] == 30.0  # halfway between 20 and 40


def test_duplicate_timestamps_are_refused(tmp_path):
    """Two rows for the same moment puts every later lag off by one."""
    duplicated = ["2024-01-01,10", "2024-01-02,20", "2024-01-02,25"]

    with pytest.raises(ValueError, match="appears twice"):
        load_series(write_csv(tmp_path, duplicated), "date", "value")


def test_out_of_order_rows_are_refused(tmp_path):
    """If the past is not behind you in the array, nothing downstream holds."""
    shuffled = ["2024-01-01,10", "2024-01-03,30", "2024-01-02,20"]

    with pytest.raises(ValueError, match="comes before the row above it"):
        load_series(write_csv(tmp_path, shuffled), "date", "value")


def test_a_value_that_is_not_a_number_is_refused(tmp_path):
    """One stray 'n/a' makes pandas read the whole column as text."""
    dirty = ["2024-01-01,10", "2024-01-02,n/a", "2024-01-03,30"]

    with pytest.raises(ValueError, match="row 1 .*: 'n/a' is not a number"):
        load_series(write_csv(tmp_path, dirty), "date", "value")


def test_a_timestamp_that_is_not_a_date_is_refused(tmp_path):
    dirty = ["2024-01-01,10", "not a date,20", "2024-01-03,30"]

    with pytest.raises(ValueError, match="is not a timestamp"):
        load_series(write_csv(tmp_path, dirty), "date", "value")


def test_a_missing_column_names_what_was_found(tmp_path):
    path = write_csv(tmp_path, CLEAN)

    with pytest.raises(ValueError, match="no column named 'cnt'"):
        load_series(path, "date", "cnt")


def test_a_single_row_has_no_spacing(tmp_path):
    with pytest.raises(ValueError, match="at least two rows"):
        load_series(write_csv(tmp_path, ["2024-01-01,10"]), "date", "value")
