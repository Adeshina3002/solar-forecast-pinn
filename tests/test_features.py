"""Unit tests for the features module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from solar_forecast.features import add_calendar_features, add_lag_features


@pytest.fixture
def hourly_df() -> pd.DataFrame:
    """200 hours of synthetic data with a known target column."""
    idx = pd.date_range("2018-01-01", periods=200, freq="h", tz="UTC")
    return pd.DataFrame({"y": np.arange(200, dtype=float)}, index=idx)


class TestLagFeatures:
    def test_lag_values_are_correct(self, hourly_df):
        out = add_lag_features(hourly_df, target_col="y", lags=(1, 24))
        # Row 100 should have lag_1=99 and lag_24=76
        assert out["y_lag_1h"].iloc[100] == 99.0
        assert out["y_lag_24h"].iloc[100] == 76.0

    def test_first_rows_are_nan(self, hourly_df):
        """The first `max(lag)` rows must have NaN in the lag column —
        no fabricated history."""
        out = add_lag_features(hourly_df, target_col="y", lags=(24,))
        assert out["y_lag_24h"].iloc[:24].isna().all()
        assert out["y_lag_24h"].iloc[24:].notna().all()

    def test_does_not_mutate_input(self, hourly_df):
        before_cols = list(hourly_df.columns)
        add_lag_features(hourly_df, target_col="y", lags=(1,))
        assert list(hourly_df.columns) == before_cols, "input was mutated"

    def test_no_future_leakage(self, hourly_df):
        """The lag at time t must come from a strictly earlier time.
        This is the data-leakage invariant — verify it directly."""
        out = add_lag_features(hourly_df, target_col="y", lags=(1, 24, 168))
        for lag_h in (1, 24, 168):
            col = f"y_lag_{lag_h}h"
            non_nan = out[col].dropna()
            for ts in non_nan.index[:50]:  # sample 50 timestamps for speed
                # The lag value must equal y at exactly `lag_h` hours earlier
                expected_ts = ts - pd.Timedelta(hours=lag_h)
                assert out.loc[ts, col] == hourly_df.loc[expected_ts, "y"]


class TestCalendarFeatures:
    def test_hour_is_circular(self, hourly_df):
        """Hour 0 and hour 24 must produce the same sin/cos values
        (otherwise the wraparound is broken)."""
        out = add_calendar_features(hourly_df)
        # First row is hour 0, row 24 is hour 0 of the next day
        assert out["hour_sin"].iloc[0] == pytest.approx(out["hour_sin"].iloc[24])
        assert out["hour_cos"].iloc[0] == pytest.approx(out["hour_cos"].iloc[24])

    def test_features_in_unit_range(self, hourly_df):
        """sin/cos outputs are in [-1, 1]."""
        out = add_calendar_features(hourly_df)
        for col in ("hour_sin", "hour_cos", "doy_sin", "doy_cos"):
            assert out[col].min() >= -1.0
            assert out[col].max() <= 1.0
