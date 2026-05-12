"""Smoke tests for XGBoostForecaster.

These are not unit tests of XGBoost itself — that library is well-tested
upstream. We just verify our wrapper preserves the Forecaster contract:
fit returns self, predict returns clipped Series at the right index, etc.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from solar_forecast.baselines import XGBoostForecaster


@pytest.fixture
def synthetic_data() -> pd.DataFrame:
    """500 hours of data where target ≈ 0.5·feature1 + 0.3·feature2 + noise."""
    rng = np.random.default_rng(0)
    n = 500
    idx = pd.date_range("2018-01-01", periods=n, freq="h", tz="UTC")
    f1 = rng.uniform(0, 1, n)
    f2 = rng.uniform(0, 1, n)
    y = np.clip(0.5 * f1 + 0.3 * f2 + rng.normal(0, 0.05, n), 0, 1)
    return pd.DataFrame({"y": y, "f1": f1, "f2": f2}, index=idx)


class TestXGBoostForecaster:
    def test_requires_feature_cols(self, synthetic_data):
        with pytest.raises(ValueError, match="feature_cols"):
            XGBoostForecaster(target_col="y").fit(synthetic_data)

    def test_predict_before_fit_raises(self, synthetic_data):
        m = XGBoostForecaster(target_col="y", feature_cols=("f1", "f2"))
        with pytest.raises(RuntimeError, match="fit"):
            m.predict(synthetic_data)

    def test_predictions_in_unit_interval(self, synthetic_data):
        """Capacity factor predictions must be clipped to [0, 1]."""
        m = XGBoostForecaster(
            target_col="y", feature_cols=("f1", "f2"),
            n_estimators=20,  # keep test fast
        ).fit(synthetic_data)
        preds = m.predict(synthetic_data)
        assert preds.min() >= 0.0
        assert preds.max() <= 1.0
        assert preds.index.equals(synthetic_data.index)

    def test_learns_signal(self, synthetic_data):
        """On data with a real signal, predictions should beat the mean."""
        m = XGBoostForecaster(
            target_col="y", feature_cols=("f1", "f2"),
            n_estimators=100,
        ).fit(synthetic_data)
        preds = m.predict(synthetic_data)
        rmse_model = ((synthetic_data["y"] - preds) ** 2).mean() ** 0.5
        rmse_mean = synthetic_data["y"].std()
        assert rmse_model < rmse_mean, "XGBoost should beat the mean predictor"

    def test_feature_importance(self, synthetic_data):
        m = XGBoostForecaster(
            target_col="y", feature_cols=("f1", "f2"),
            n_estimators=50,
        ).fit(synthetic_data)
        imp = m.feature_importance
        assert imp is not None
        assert set(imp.index) == {"f1", "f2"}
        # f1 has weight 0.5, f2 has weight 0.3 — f1 should be more important
        assert imp["f1"] > imp["f2"]
