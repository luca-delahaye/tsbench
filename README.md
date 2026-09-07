# tsbench

Score forecasting models against a naive baseline, under walk-forward
validation that cannot accidentally cheat.

## What this is

Take any column of numbers recorded over time — bike rentals, electricity
load, order volumes — and find out whether a model actually beats doing
nothing clever.

**The models are the easy part.** Most of them are three lines. The product
is the harness around them: the machinery that evaluates a forecast in a way
that cannot see the future. Anyone can call a forecasting library. The
harder thing is being able to show that your evaluation is honest, which is
what this repository is for.

## Install

```bash
git clone git@github.com:luca-delahaye/tsbench.git
cd tsbench
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Getting the data

The example data is not committed — it belongs to UCI, not to this
repository. Two files, about 1 MB:

```bash
mkdir -p data
curl -L -o data/bike.zip \
    https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip
unzip -o data/bike.zip -d data/
```

That gives `data/day.csv` (731 daily rows, no gaps) and `data/hour.csv`
(17,379 hourly rows, with gaps). Any other CSV with a timestamp column and a
value column works just as well.

## Quickstart

Any CSV with a timestamp column and a value column:

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

## How the evaluation works

A random train/test split is meaningless for a time series: to predict
Wednesday, the model would be handed Tuesday *and Thursday*. In real life
the future is not available, so a test that supplies it measures nothing.

Instead the origin walks forward. Train on everything up to a cut point,
predict the next few steps, score, then move the cut and repeat:

```
|---- train ----|-test-|
|------ train ------|-test-|
|-------- train --------|-test-|
|---------- train ----------|-test-|
```

Two parameters: `--horizon` is how far ahead each fold predicts, `--step` is
how far the cut advances between folds. Step defaults to horizon, so the
test blocks tile exactly — nothing is scored twice and nothing is skipped.

**Every model is scored on identical folds.** The first cut point is the
largest `min_train_size` in the run, so adding a seasonal model with period
24 pushes every model's first cut forward together. Without that, MASE would
be a ratio of two errors measured on different data, which is not a
comparison at all.

Incomplete folds are never emitted. A short final fold would be scored over
fewer points than the others, so averaging it with them is arithmetic
nonsense.

## Metrics

| Metric | Meaning |
|---|---|
| **MAE** | Average size of a miss, in the series' own units |
| **RMSE** | Same, but one big miss costs more than several small ones. RMSE near MAE means steadily mediocre; RMSE far above it means usually fine with rare disasters |
| **MASE** | The model's MAE divided by the baseline's, on the same test block. Below 1 beats the baseline |

MASE is the headline because it is the only one whose value means something
on its own. A MAE of 12.4 is excellent on a series averaging 12,000 and
dreadful on one averaging 120.

**Note on the definition.** This is the *relative* form: a ratio of two
out-of-sample errors, both measured over the same block at the same horizon.
Hyndman's MASE instead scales by the one-step naive error measured on the
training data. Those are different numbers and the two are not comparable.
The relative form is used here because it keeps the horizons matched — if
the model is asked to predict 24 hours ahead, so is the baseline it is
measured against.

Both the mean and the median MASE are reported, along with the share of
folds the model actually won, because one number per model would be the
easiest thing in the world to mislead with.

## Results

Capital Bikeshare, Washington DC, 2011–2012
([UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)).
The same system at two sampling rates.

### Daily, horizon 7 — nothing beats naive

```
model                     MAE     RMSE     MASE   median    won
Naive                   921.0   1103.9   1.0000   1.0000    0%
Drift                   934.1   1119.7   1.0252   1.0026   47%
SeasonalNaive(7)        934.3   1193.0   1.1943   1.0749   45%
Mean                   1612.9   1729.5   2.1917   1.5478   27%
```

731 points, 103 folds. **Nothing beat the baseline.** "Tomorrow will be like
today" is hard to improve on when the weekly cycle is weak — the daily
totals only vary about 10% across weekdays.

Note what the spread shows that the mean hides. Seasonal naive has a mean
MASE of 1.19, which reads as clearly worse. But it beats naive in **45% of
folds**: it is close to a coin flip that occasionally goes badly wrong, not
a model that is uniformly bad. That is a different fact, and only visible
because the distribution is reported.

### Hourly, horizon 24 — the seasonal baseline wins decisively

```
model                     MAE     RMSE     MASE   median    won
SeasonalNaive(24)        63.7     88.6   0.5449   0.3752   90%
Drift                   140.4    183.2   0.9996   0.9996   93%
Naive                   140.5    183.3   1.0000   1.0000    0%
Mean                    131.7    167.4   1.0137   0.9266   72%
```

17,544 points, 730 folds. The daily cycle here is enormous — about 12
rentals at 3am against 461 at 5pm — so "the same hour yesterday" cuts the
error roughly in half and wins 90% of folds.

**Caveat, stated rather than buried.** The hourly file is missing 165 of its
17,544 rows, in 75 separate gaps. This run used `--fill-gaps`, so those rows
were interpolated, and some test blocks therefore contain invented numbers
that every model is scored against. The daily result above needs no such
flag and is the cleaner of the two.

Reproducing the hourly run needs the timestamp assembled first, since the
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

### Reading these two together

The daily result is the honest negative: most of the time nothing beats the
baseline, and a tool that never says so is not measuring anything. The
hourly result shows the same harness detecting a real signal when one is
there. Neither is rigged in either direction.

## The lookahead test

Lookahead bias — code that saw data from after the cut point — is how most
amateur backtests quietly produce results that cannot be reproduced in real
time. It almost never happens at the split. It happens in feature
construction, and every version of it is correct-looking code: nothing
errors, nothing looks wrong on review, and the result is simply better than
it should be.

Being careful is not a strategy. So there is a test:

> Score a clean prefix of the series. Then score a longer series whose first
> N points are identical and whose remaining points have been replaced with
> garbage. Every fold living entirely inside the clean part must produce
> **identical** scores. Not similar — identical.

Those folds were never supposed to see the corrupted region. If one digit
moves, something read the future. This catches a centred rolling window, an
off-by-one in a trailing one, and a scaler fitted before splitting, without
caring how careful anyone was being.

A second test corrupts a point in the *past* instead, and requires the
scores to change — because a test that cannot fail proves nothing.

The guarantee itself lives in two lines, in the only file that slices the
series:

```python
train = values[fold.train_start:fold.train_end]
actual = values[fold.test_start:fold.test_end]
```

Everything else receives arrays and never learns where they came from.

## Reading a CSV is not a formality

Every module downstream is built on one assumption:

> position N is exactly N steps in time

That is a lie unless somebody checks it. If three hours are missing from an
hourly file, "look back 24 positions" quietly lands 27 hours ago, forever,
and nothing tells you. So `data.py` is an airlock, not a converter. It
refuses unparseable dates and numbers, duplicate or out-of-order timestamps,
and gaps — naming the row and the timestamp:

```
error: row 29: 1 missing at 2011-01-02 04:00:00 (jumps to 2011-01-02 06:00:00);
75 gaps in total. Pass --fill-gaps to interpolate them.
```

Gaps are refused **by default**. Interpolating a hole and then testing over
it scores the model against numbers we invented, and it will do beautifully.
`--fill-gaps` exists, but it has to be typed — and it reports how many rows
it added.

## What this deliberately does not do

- Multivariate series or exogenous regressors. The bike data ships with
  temperature and humidity; they are ignored on purpose.
- Deep learning. Nothing here needs it.
- Hyperparameter search.
- Irregular or mixed-frequency series.
- A web interface.

Shipping something small that is correct beats shipping something large that
is unfinished.

## Layout

```
tsbench/
  data.py       CSV in, validated series out. The only file that knows dates
  splits.py     pure arithmetic: which positions to train and test on
  models.py     one interface, four baselines
  metrics.py    MAE, RMSE, MASE
  evaluate.py   the loop, and the only file that slices the series
  cli.py        python -m tsbench run data.csv
tests/
```

The arrows only point one way. `data.py` does not know models exist,
`models.py` does not know CSV files exist, and `metrics.py` just subtracts
two arrays.

## Tests

```bash
python -m pytest
```

68 tests. Expected values are worked out by hand rather than recomputed from
the implementation, since a test that repeats the implementation's own
arithmetic cannot catch that arithmetic being wrong.

## Next

- Ridge regression on lag features, as the first model that is not a baseline
- Prediction intervals from residual quantiles
- Direct versus recursive multi-step forecasting
