"""Tests for the command line.

The CLI decides nothing, so these check the two small choices it makes and
one run from end to end.
"""

import pandas as pd
import pytest

from tsbench.cli import build_models, default_period, main


def test_period_is_read_from_the_spacing():
    """Daily rows repeat weekly, hourly rows repeat daily."""
    assert default_period(pd.Timedelta(days=1)) == 7
    assert default_period(pd.Timedelta(hours=1)) == 24


def test_an_unfamiliar_spacing_gets_no_seasonal_model():
    """No guess is better than a wrong one."""
    assert default_period(pd.Timedelta(minutes=17)) is None
    assert len(build_models(None)) == 3


def test_a_known_period_adds_seasonal_naive():
    names = [model.name for model in build_models(24)]
    assert "SeasonalNaive(24)" in names
    assert len(names) == 4


def write_series(tmp_path, days=120):
    """A series with a weekly cycle, written as a CSV the CLI can read."""
    stamps = pd.date_range("2024-01-01", periods=days, freq="D")
    values = [100 + 10 * (i % 7) + i for i in range(days)]
    path = tmp_path / "series.csv"
    pd.DataFrame({"date": stamps, "value": values}).to_csv(path, index=False)
    return path


def test_end_to_end(tmp_path, capsys):
    path = write_series(tmp_path)
    main(["run", str(path), "--time", "date", "--value", "value", "--horizon", "7"])

    printed = capsys.readouterr().out
    assert "120 points, daily" in printed
    for name in ("Naive", "Mean", "Drift", "SeasonalNaive(7)"):
        assert name in printed


def test_a_bad_column_exits_with_a_message(tmp_path):
    path = write_series(tmp_path)
    with pytest.raises(SystemExit, match="no column named 'nope'"):
        main(["run", str(path), "--time", "date", "--value", "nope"])


def test_a_gap_exits_and_suggests_the_flag(tmp_path):
    """The message must name the flag, not the Python keyword."""
    frame = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-04"],
        "value": [10, 20, 40],
    })
    path = tmp_path / "gapped.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(SystemExit, match=r"--fill-gaps"):
        main(["run", str(path), "--time", "date", "--value", "value"])
