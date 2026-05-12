"""
Time-series feature engineering: lagged values, rolling statistics, calendar
features. Distinct from physics.py — those features come from first principles,
these come from the temporal structure of the data itself.

Key design note: these functions operate on the *full* dataframe before the
train/val/test split. That's intentional and *not* data leakage for lag
features specifically: a lag feature at time t uses the value at time t - k,
which is always strictly in the past relative to t. Computing lags on the
full series and then splitting gives the *same* values you'd compute by
splitting first and joining the right history at evaluation time — but
without the bookkeeping. The deployment scenario (forecasting hour t with
access to hours t-1, t-24, t-168) is exactly preserved.

If you ever add features that DO use future information (e.g. forward-looking
moving averages, target-encoded categorical means), they must move into the
preprocessing step that runs *after* splitting. That distinction is the
single most important data-leakage rule in time-series ML.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


DEFAULT_LAGS = (1, 24, 168)  # 1h, 1d, 1w — the standard "look back" windows


def add_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lags: Iterable[int] = DEFAULT_LAGS,
) -> pd.DataFrame:
    """
    Add lagged copies of `target_col` to the dataframe.

    Output columns are named `{target_col}_lag_{n}h`. The first `max(lags)`
    rows will contain NaN in the lag columns — these are not dropped here
    because the model fit() method handles dropping consistently across
    feature sets.

    Why these specific lags by default:
        lag 1   — the persistence signal; the strongest single feature in solar
        lag 24  — same hour yesterday; captures diurnal+weather persistence
        lag 168 — same hour-of-week last week; captures weekly periodicity
                  (e.g. industrial load patterns, though for *generation* this
                  matters less than for *load*)
    """
    df = df.copy()
    for lag in lags:
        col_name = f"{target_col}_lag_{lag}h"
        df[col_name] = df[target_col].shift(lag)
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sin/cos encodings of hour-of-day and day-of-year.

    Why sin/cos rather than raw hour=0..23 / dayofyear=1..365: the model
    needs to know that hour 23 is adjacent to hour 0 (midnight wraps), and
    that day 365 is adjacent to day 1. Raw integer features can't express
    this; sin/cos pairs do, because (sin θ, cos θ) gives every point on the
    unit circle a unique 2D coordinate that respects wraparound.

    These features are cheap and almost always help tree-based models —
    XGBoost can learn the diurnal cycle from them with much less data than
    it would need to learn it from the raw timestamp alone.
    """
    df = df.copy()
    import numpy as np
    hour = df.index.hour
    doy = df.index.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df
