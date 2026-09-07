# tsbench

Walk-forward evaluation of forecasting models against a naive baseline.

## What it does

Reads a CSV containing a time series, splits it using rolling-origin
validation, runs several forecasting models over the same folds, and reports
how each one scored relative to a naive baseline.

The models included are baselines: naive, seasonal naive, mean and drift.
Most of the code is the evaluation harness rather than the models — the
splitting, the input validation and the scoring.

## Install

```bash
git clone git@github.com:luca-delahaye/tsbench.git
cd tsbench
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Getting the data

The example data is not committed, since it belongs to UCI. Two files, about
1 MB:

```bash
mkdir -p data
curl -L -o data/bike.zip \
    https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip
unzip -o data/bike.zip -d data/
```

That gives `data/day.csv` (731 daily rows, no gaps) and `data/hour.csv`
(17,379 hourly rows, with gaps). Any CSV with a timestamp column and a value
column will work.

## Quickstart

```bash
python -m tsbench run data/day.csv --time dteday --value cnt --horizon 7
```

```
data/day.csv: 731 points, daily, 2011-01-01 to 2012-12-31
horizon 7, step 7, 103 folds, first cut at 7

model                     MAE     RMSE     MASE   median    won
---------------------------------------------------------------
Naive                   921.0   1103.9   1.0000   1.0000    0%
Drift                   934.1   1119.7   1.0252   1.0026   47%
SeasonalNaive(7)        934.3   1193.0   1.1943   1.0749   45%
Mean                   1612.9   1729.5   2.1917   1.5478   27%
```

| Flag | Meaning |
|---|---|
| `--time`, `--value` | column names in the CSV |
| `--horizon` | steps predicted per fold (default 7) |
| `--step` | how far the cut advances between folds (default: horizon) |
| `--period` | seasonal cycle length (default: inferred from the spacing) |
| `--fill-gaps` | interpolate missing rows instead of refusing the file |
| `--plot` | write a PNG of the series and the final test block |

## How the evaluation works

A random train/test split does not measure forecasting performance on a time
series: predicting Wednesday would give the model access to both Tuesday and
Thursday. Since later values are not available when a forecast is actually
made, a test that supplies them measures something else.

Instead the origin moves forward. The model trains on everything up to a cut
point, predicts the next few steps, is scored, and the cut advances:

```
|---- train ----|-test-|
|------ train ------|-test-|
|-------- train --------|-test-|
|---------- train ----------|-test-|
```

`--horizon` is how far ahead each fold predicts. `--step` is how far the cut
advances between folds, and defaults to the horizon so that test blocks tile
exactly: no position is scored twice and none is skipped.

All models are scored on the same folds. The first cut point is the largest
`min_train_size` among the models in the run, so adding a seasonal model with
period 24 moves the first cut forward for every model. If the folds differed,
MASE would be a ratio of two errors measured on different data.

Incomplete folds are not produced. A short final fold would be scored over
fewer points than the others, so it could not be averaged with them.

## Metrics

| Metric | Meaning |
|---|---|
| **MAE** | Average size of an error, in the series' units |
| **RMSE** | Same, weighted so one large error costs more than several small ones. RMSE close to MAE indicates consistent errors; RMSE much larger indicates occasional large ones |
| **MASE** | The model's MAE divided by the baseline's, over the same test block. Below 1 means the model beat the baseline |

MASE is the headline figure because its value can be read without knowing the
scale of the series. An MAE of 12.4 means something different on a series
averaging 12,000 than on one averaging 120.

**Definition used.** This is the relative form: a ratio of two out-of-sample
errors, both measured over the same block at the same horizon. Hyndman's MASE
scales instead by the one-step naive error measured on the training data.
The two are different quantities and are not comparable. The relative form is
used here so that the model and the baseline are asked the same question — if
the model predicts 24 hours ahead, so does the baseline it is compared to.

The mean MASE, the median MASE and the share of folds won are all reported, so
that a single averaged figure does not stand in for the distribution.

## Results

Capital Bikeshare, Washington DC, 2011–2012
([UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)),
at two sampling rates.

### Daily, horizon 7

```
model                     MAE     RMSE     MASE   median    won
Naive                   921.0   1103.9   1.0000   1.0000    0%
Drift                   934.1   1119.7   1.0252   1.0026   47%
SeasonalNaive(7)        934.3   1193.0   1.1943   1.0749   45%
Mean                   1612.9   1729.5   2.1917   1.5478   27%
```

731 points, 103 folds. No model beat the baseline. The weekly cycle in the
daily totals is small — the weekday averages range from 4,229 to 4,690, about
10% — so repeating the previous value is difficult to improve on.

The mean and the spread disagree here. Seasonal naive has a mean MASE of
1.1943, but beats naive in 45% of folds and has a median of 1.0749. Its
average is raised by a small number of poor folds rather than by being
consistently worse.

### Hourly, horizon 24

```
model                     MAE     RMSE     MASE   median    won
SeasonalNaive(24)        63.7     88.6   0.5449   0.3752   90%
Drift                   140.4    183.2   0.9996   0.9996   93%
Naive                   140.5    183.3   1.0000   1.0000    0%
Mean                    131.7    167.4   1.0137   0.9266   72%
```

17,544 points, 730 folds. The daily cycle is large — hourly averages run from
about 12 rentals at 3am to 461 at 5pm — and seasonal naive roughly halves the
error, winning 90% of folds.

**Caveat.** The hourly file is missing 165 of its 17,544 rows, across 75
gaps. This run used `--fill-gaps`, so those rows were interpolated and some
test blocks contain values that were not measured. The daily result requires
no such flag.

Reproducing the hourly run requires assembling the timestamp first, since the
raw file splits it across two columns:

```python
import pandas as pd
h = pd.read_csv("data/hour.csv")
h["timestamp"] = pd.to_datetime(h["dteday"]) + pd.to_timedelta(h["hr"], unit="h")
h[["timestamp", "cnt"]].to_csv("data/hour_prepared.csv", index=False)
```

```bash
python -m tsbench run data/hour_prepared.csv --time timestamp --value cnt \
    --horizon 24 --fill-gaps
```

### Summary

On the daily series no model beat the baseline. On the hourly series, where
there is a strong repeating cycle, the seasonal baseline beat it clearly. The
two results come from the same code and the same underlying system.

## The lookahead test

Lookahead bias means code that used data from after the cut point. It is
usually introduced during feature construction rather than at the split, and
it does not raise an error: the code runs, and the reported score is better
than it should be.

`tests/test_evaluate.py` checks for it directly:

> Score a prefix of the series. Then score a longer series whose first N
> points are identical and whose remaining points have been replaced with
> large arbitrary values. Every fold contained entirely within the first N
> points must produce identical scores.

Those folds were not supposed to read the replaced region, so any change in
the score means something did. This covers a centred rolling window, an
off-by-one in a trailing window, and a scaler fitted before splitting, without
requiring the code to be inspected for them.

A second test replaces a point in the *past* instead and requires the scores
to change, so that the first test is known to be capable of failing.

The property being tested comes from `evaluate.py` being the only module that
slices the series:

```python
train = values[fold.train_start:fold.train_end]
actual = values[fold.test_start:fold.test_end]
```

Every other module receives arrays and has no access to their position in the
series.

## Input validation

Every module downstream of `data.py` assumes that position N is exactly N
steps in time. If rows are missing from an hourly file, looking back 24
positions lands more than 24 hours earlier, and nothing reports it.

`data.py` therefore refuses unparseable dates and values, duplicate or
out-of-order timestamps, and gaps, identifying the row and the timestamp:

```
error: row 29: 1 missing at 2011-01-02 04:00:00 (jumps to 2011-01-02 06:00:00);
75 gaps in total. Pass --fill-gaps to interpolate them.
```

Gaps are refused by default. Interpolating a gap and then evaluating over it
scores the model partly against interpolated values. `--fill-gaps` enables it
explicitly and reports how many rows were added.

## Out of scope

- Multivariate series and exogenous regressors. The bike data includes
  temperature and humidity; they are not used.
- Neural network models.
- Hyperparameter search.
- Irregular or mixed-frequency series.
- A web interface.

## Layout

```
tsbench/
  data.py       CSV in, validated series out; the only module using pandas
  splits.py     which positions to train and test on, as arithmetic
  models.py     one interface, four baselines
  metrics.py    MAE, RMSE, MASE
  evaluate.py   the fold loop; the only module that slices the series
  cli.py        python -m tsbench run data.csv
tests/
```

Dependencies run one way. `data.py` does not import models, `models.py` does
not read files, and `metrics.py` operates on two arrays.

## Tests

```bash
python -m pytest
```

68 tests. Expected values are written out by hand rather than recomputed from
the implementation, so that a test cannot agree with an incorrect
implementation by repeating its arithmetic.

## Possible extensions

- Ridge regression on lag features, as a first non-baseline model
- Prediction intervals from residual quantiles
- Direct versus recursive multi-step forecasting
