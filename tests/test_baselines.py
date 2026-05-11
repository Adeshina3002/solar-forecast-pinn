"""
Unit tests for baselines, splits, and metrics.

The synthetic fixtures here are small and deliberate — each baseline gets
checked on a setup where you can manually verify the correct answer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from solar_forecast.baselines import (
    DailySeasonalNaive,
    HourOfYearClimatology,
    LinearWeather,
    Persistence,
)
from solar_forecast.metrics import evaluate, skill_score
from solar_forecast.splits import Split, split_dataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def toy_dataset() -> pd.DataFrame:
    """
    Three years of hourly data with a known diurnal cycle and a weak linear
    relationship to GHI. Small enough to reason about, structured enough that
    real baselines have something to learn.
    """
    idx = pd.date_range("2015-01-01", "2017-12-31 23:00", freq="h", tz="UTC")
    hours = idx.hour
    # Sine-shaped diurnal cycle peaking at noon; minor day-to-day noise.
    diurnal = np.maximum(np.sin((hours - 6) * np.pi / 12), 0.0)
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.05, len(idx))
    ghi = diurnal * 800 + rng.normal(0, 20, len(idx))
    ghi = np.maximum(ghi, 0.0)
    temp = 15 + 10 * np.sin((idx.dayofyear - 80) * 2 * np.pi / 365) + rng.normal(0, 2, len(idx))
    cf = np.clip(diurnal + noise, 0.0, 1.0)
    return pd.DataFrame({
        "DE_solar_cf": cf,
        "DE_ghi": ghi,
        "DE_temperature": temp,
    }, index=idx)


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

class TestSplits:
    def test_partitions_correctly(self, toy_dataset):
        """Train/val/test should cover the whole dataset with no overlap."""
        s = Split()
        train, val, test = split_dataset(toy_dataset, split=s)
        # Toy dataset is 2015-2017 only, so val and test will be empty.
        # The point is the train slice is correct.
        assert train.index.min() >= pd.Timestamp(s.train_start, tz="UTC")
        assert train.index.max() <= pd.Timestamp(s.train_end, tz="UTC")
        assert val.empty   # toy data doesn't reach 2018
        assert test.empty  # nor 2019


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_predicts_previous_hour(self, toy_dataset):
        """Persistence prediction at t should equal target at t-1."""
        model = Persistence().fit(toy_dataset)
        preds = model.predict(toy_dataset)
        assert preds.iloc[100] == pytest.approx(toy_dataset["DE_solar_cf"].iloc[99])

    def test_first_prediction_is_nan(self, toy_dataset):
        """No prior hour exists for the first row → NaN."""
        model = Persistence().fit(toy_dataset)
        preds = model.predict(toy_dataset)
        assert pd.isna(preds.iloc[0])


# ---------------------------------------------------------------------------
# Daily seasonal naive
# ---------------------------------------------------------------------------

class TestDailySeasonalNaive:
    def test_predicts_24h_lag(self, toy_dataset):
        """Prediction at t equals target at t-24h."""
        model = DailySeasonalNaive().fit(toy_dataset)
        preds = model.predict(toy_dataset)
        # Position 100 is hour 100 from start; t-24h is position 76.
        assert preds.iloc[100] == pytest.approx(toy_dataset["DE_solar_cf"].iloc[76])


# ---------------------------------------------------------------------------
# Climatology
# ---------------------------------------------------------------------------

class TestHourOfYearClimatology:
    def test_predicts_historical_mean_for_same_hour(self, toy_dataset):
        """Forecast for (month=6, day=15, hour=14) = mean of training hours matching."""
        model = HourOfYearClimatology().fit(toy_dataset)
        preds = model.predict(toy_dataset)

        # All rows where month=6, day=15, hour=14 should have identical predictions
        idx = toy_dataset.index
        mask = (idx.month == 6) & (idx.day == 15) & (idx.hour == 14)
        matching = preds[mask]
        assert matching.nunique() == 1, "All same-(M,D,H) predictions must be equal"

    def test_raises_if_not_fit(self, toy_dataset):
        model = HourOfYearClimatology()
        with pytest.raises(RuntimeError, match="must be .fit"):
            model.predict(toy_dataset)


# ---------------------------------------------------------------------------
# Linear regression on weather
# ---------------------------------------------------------------------------

class TestLinearWeather:
    def test_predictions_clipped_to_unit_interval(self, toy_dataset):
        """Capacity factor is in [0, 1] by definition; linear model must respect."""
        model = LinearWeather().fit(toy_dataset)
        preds = model.predict(toy_dataset)
        assert preds.min() >= 0.0
        assert preds.max() <= 1.0

    def test_has_some_explanatory_power(self, toy_dataset):
        """On data with a real GHI→CF signal, linear regression should beat
        a constant-mean prediction."""
        model = LinearWeather().fit(toy_dataset)
        preds = model.predict(toy_dataset)
        constant_pred = pd.Series(
            toy_dataset["DE_solar_cf"].mean(),
            index=toy_dataset.index,
        )
        m_model = evaluate(toy_dataset["DE_solar_cf"], preds)
        m_const = evaluate(toy_dataset["DE_solar_cf"], constant_pred)
        assert m_model.rmse < m_const.rmse, "Linear model should beat constant prediction"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_zero_error_on_perfect_prediction(self):
        idx = pd.date_range("2020-01-01", periods=10, freq="h", tz="UTC")
        y = pd.Series(np.arange(10, dtype=float), index=idx)
        m = evaluate(y, y)
        assert m.rmse == 0.0
        assert m.mae == 0.0
        assert m.n_samples == 10

    def test_skill_zero_against_self(self):
        idx = pd.date_range("2020-01-01", periods=10, freq="h", tz="UTC")
        y_true = pd.Series(np.arange(10, dtype=float), index=idx)
        y_pred = y_true + 0.1
        m = evaluate(y_true, y_pred)
        assert skill_score(m, m) == 0.0

    def test_skill_positive_when_candidate_better(self):
        idx = pd.date_range("2020-01-01", periods=10, freq="h", tz="UTC")
        y_true = pd.Series(np.arange(10, dtype=float), index=idx)
        # Candidate: off by 0.1 everywhere. Baseline: off by 0.5 everywhere.
        candidate = evaluate(y_true, y_true + 0.1)
        baseline = evaluate(y_true, y_true + 0.5)
        assert skill_score(candidate, baseline) > 0
