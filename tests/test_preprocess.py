"""
Unit tests for the preprocessing pipeline.

Each test uses small synthetic fixtures rather than the real OPSD data — fast
to run, and it makes the *expected behavior* explicit. If a real-data test
fails you know preprocessing broke; if a synthetic test fails you know
*which preprocessing decision* broke.

Run with: pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from solar_forecast.preprocess import (
    PreprocessConfig,
    add_derived_features,
    align_to_window,
    clip_nighttime_irradiance,
    clip_nighttime_solar,
    interpolate_short_gaps,
    normalize_to_capacity_factor,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def hourly_ts() -> pd.DatetimeIndex:
    """Two full days of hourly UTC timestamps for synthetic tests."""
    return pd.date_range("2018-06-21", periods=48, freq="h", tz="UTC")


@pytest.fixture
def fake_solar(hourly_ts) -> pd.DataFrame:
    """
    Synthetic solar generation with two known anomalies:
    - 02:00 UTC on day 1: 50 MW (should be 0, it's the middle of the night)
    - 14:00 UTC on day 1: 25000 MW (legitimate midday peak)
    """
    df = pd.DataFrame({
        "utc_timestamp": hourly_ts,
        "DE_solar_generation_actual": 0.0,
        "DE_solar_capacity": 45000.0,
    })
    df.loc[2, "DE_solar_generation_actual"] = 50.0       # nighttime artifact
    df.loc[14, "DE_solar_generation_actual"] = 25000.0   # legit daytime
    df.loc[15, "DE_solar_generation_actual"] = 27000.0   # legit daytime
    return df


# ---------------------------------------------------------------------------
# clip_nighttime_solar
# ---------------------------------------------------------------------------

class TestClipNighttimeSolar:
    def test_zeros_out_nighttime_anomaly(self, fake_solar):
        """A 50 MW reading at 02:00 UTC should be clipped to zero."""
        cleaned = clip_nighttime_solar(
            fake_solar,
            solar_col="DE_solar_generation_actual",
            threshold_mw=1.0,
        )
        assert cleaned.loc[2, "DE_solar_generation_actual"] == 0.0

    def test_preserves_legitimate_daytime(self, fake_solar):
        """Daytime values should pass through untouched."""
        cleaned = clip_nighttime_solar(
            fake_solar,
            solar_col="DE_solar_generation_actual",
            threshold_mw=1.0,
        )
        assert cleaned.loc[14, "DE_solar_generation_actual"] == 25000.0
        assert cleaned.loc[15, "DE_solar_generation_actual"] == 27000.0

    def test_does_not_mutate_input(self, fake_solar):
        """Function should return a new DataFrame, not modify the caller's copy."""
        before = fake_solar.loc[2, "DE_solar_generation_actual"]
        clip_nighttime_solar(fake_solar, solar_col="DE_solar_generation_actual", threshold_mw=1.0)
        after = fake_solar.loc[2, "DE_solar_generation_actual"]
        assert before == after, "input was mutated"

    def test_threshold_respected(self, fake_solar):
        """A value below the threshold should not be clipped."""
        fake_solar.loc[2, "DE_solar_generation_actual"] = 0.5
        cleaned = clip_nighttime_solar(
            fake_solar,
            solar_col="DE_solar_generation_actual",
            threshold_mw=1.0,
        )
        assert cleaned.loc[2, "DE_solar_generation_actual"] == 0.5


# ---------------------------------------------------------------------------
# clip_nighttime_irradiance
# ---------------------------------------------------------------------------

class TestClipNighttimeIrradiance:
    def test_clips_both_components(self, hourly_ts):
        """Direct and diffuse both clipped when GHI exceeds threshold at night."""
        df = pd.DataFrame({
            "utc_timestamp": hourly_ts,
            "direct": 0.0,
            "diffuse": 0.0,
        })
        df.loc[2, ["direct", "diffuse"]] = [3.0, 2.0]   # 5 W/m² at 02:00 UTC
        df.loc[14, ["direct", "diffuse"]] = [600.0, 200.0]  # legit daytime

        cleaned = clip_nighttime_irradiance(
            df,
            direct_col="direct",
            diffuse_col="diffuse",
            threshold_w_m2=1.0,
        )
        assert cleaned.loc[2, "direct"] == 0.0
        assert cleaned.loc[2, "diffuse"] == 0.0
        assert cleaned.loc[14, "direct"] == 600.0


# ---------------------------------------------------------------------------
# normalize_to_capacity_factor
# ---------------------------------------------------------------------------

class TestCapacityFactor:
    def test_basic_division(self, fake_solar):
        out = normalize_to_capacity_factor(
            fake_solar,
            generation_col="DE_solar_generation_actual",
            capacity_col="DE_solar_capacity",
            output_col="cf",
        )
        # 25000 / 45000 ≈ 0.556
        assert out.loc[14, "cf"] == pytest.approx(25000 / 45000, abs=1e-6)

    def test_clips_above_one(self):
        """If reported generation exceeds capacity, clip to 1.0."""
        df = pd.DataFrame({
            "DE_solar_generation_actual": [50000.0],   # > capacity
            "DE_solar_capacity": [45000.0],
        })
        out = normalize_to_capacity_factor(
            df,
            generation_col="DE_solar_generation_actual",
            capacity_col="DE_solar_capacity",
            output_col="cf",
        )
        assert out.loc[0, "cf"] == 1.0

    def test_clips_below_zero(self):
        """Negative generation (sensor drift) clipped to 0."""
        df = pd.DataFrame({
            "DE_solar_generation_actual": [-5.0],
            "DE_solar_capacity": [45000.0],
        })
        out = normalize_to_capacity_factor(
            df,
            generation_col="DE_solar_generation_actual",
            capacity_col="DE_solar_capacity",
            output_col="cf",
        )
        assert out.loc[0, "cf"] == 0.0

    def test_nan_capacity_propagates(self):
        """If capacity is unknown for an hour, capacity factor is undefined."""
        df = pd.DataFrame({
            "DE_solar_generation_actual": [1000.0],
            "DE_solar_capacity": [np.nan],
        })
        out = normalize_to_capacity_factor(
            df,
            generation_col="DE_solar_generation_actual",
            capacity_col="DE_solar_capacity",
            output_col="cf",
        )
        assert pd.isna(out.loc[0, "cf"])


# ---------------------------------------------------------------------------
# align_to_window
# ---------------------------------------------------------------------------

class TestAlignToWindow:
    def test_fills_missing_hours_with_nan(self):
        """A gap in the source becomes an explicit NaN row, not silent absence."""
        df = pd.DataFrame({
            "utc_timestamp": pd.to_datetime([
                "2018-06-21 00:00:00",
                "2018-06-21 02:00:00",   # missing 01:00
            ], utc=True),
            "value": [10.0, 20.0],
        })
        aligned = align_to_window(df, start="2018-06-21 00:00", end="2018-06-21 02:00")
        assert len(aligned) == 3
        assert pd.isna(aligned.iloc[1]["value"])

    def test_respects_window(self):
        """Data outside [start, end] is dropped."""
        df = pd.DataFrame({
            "utc_timestamp": pd.to_datetime([
                "2017-12-31 23:00:00",   # before window
                "2018-06-21 00:00:00",   # in window
                "2020-01-01 00:00:00",   # after window
            ], utc=True),
            "value": [1.0, 2.0, 3.0],
        })
        aligned = align_to_window(df, start="2018-01-01", end="2019-12-31 23:00")
        assert 2.0 in aligned["value"].values
        assert 1.0 not in aligned["value"].dropna().values
        assert 3.0 not in aligned["value"].dropna().values


# ---------------------------------------------------------------------------
# interpolate_short_gaps
# ---------------------------------------------------------------------------

class TestInterpolation:
    def test_fills_one_hour_gap(self):
        """A single NaN flanked by values gets linearly interpolated."""
        idx = pd.date_range("2018-06-21", periods=5, freq="h", tz="UTC")
        df = pd.DataFrame({"x": [10.0, np.nan, 30.0, 40.0, 50.0]}, index=idx)
        out = interpolate_short_gaps(df, columns=["x"], max_gap_hours=3)
        assert out.iloc[1]["x"] == pytest.approx(20.0)

    def test_leaves_long_gaps_alone(self):
        """Gaps longer than max_gap_hours stay NaN — we don't fabricate data."""
        idx = pd.date_range("2018-06-21", periods=10, freq="h", tz="UTC")
        values = [10.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 100.0]
        df = pd.DataFrame({"x": values}, index=idx)
        out = interpolate_short_gaps(df, columns=["x"], max_gap_hours=3)
        # The 8-hour gap is longer than max_gap_hours=3, so it stays NaN
        assert out["x"].isna().sum() == 8


# ---------------------------------------------------------------------------
# add_derived_features
# ---------------------------------------------------------------------------

class TestDerivedFeatures:
    def test_ghi_is_sum(self):
        """GHI = direct + diffuse, by definition."""
        df = pd.DataFrame({
            "DE_radiation_direct_horizontal": [500.0],
            "DE_radiation_diffuse_horizontal": [200.0],
            "IT_radiation_direct_horizontal": [600.0],
            "IT_radiation_diffuse_horizontal": [150.0],
        })
        out = add_derived_features(df)
        assert out.loc[0, "DE_ghi"] == 700.0
        assert out.loc[0, "IT_ghi"] == 750.0


# ---------------------------------------------------------------------------
# PreprocessConfig
# ---------------------------------------------------------------------------

class TestConfig:
    def test_is_frozen(self):
        """Mutating config should raise — prevents a class of bug."""
        cfg = PreprocessConfig()
        with pytest.raises((AttributeError, Exception)):  # FrozenInstanceError subclass
            cfg.overlap_start = "2020-01-01"  # type: ignore[misc]
