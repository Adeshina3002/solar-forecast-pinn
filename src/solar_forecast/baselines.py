"""
Baseline forecasters for solar capacity factor.

Each baseline implements the same interface:
    .fit(df) — learn whatever the baseline needs from training data
    .predict(df) — produce hourly forecasts for the index of `df`

This matches the scikit-learn convention deliberately. The XGBoost baseline,
the LSTM, and the PINN will all expose the same two methods. That uniformity
is what lets us write one evaluation loop that works for every model.

The four baselines, in increasing sophistication:

    1. Persistence            — y_hat(t) = y(t - 1h). No fitting required.
    2. DailySeasonalNaive     — y_hat(t) = y(t - 24h). The hardest to beat.
    3. HourOfYearClimatology  — y_hat(t) = mean over training data of the same
                                (month, day, hour) tuple. Pure seasonality.
    4. LinearWeather          — y_hat(t) = a·GHI(t) + b·T(t) + c. Simplest model
                                that uses weather. The gateway baseline.

Notes on persistence-style baselines: they need "lookback" data — the value at
t-1h or t-24h. At inference, that data must be the *true* historical value
(an oracle), not a model prediction (a recursive forecast). We are forecasting
hour-ahead, given perfect knowledge of recent history.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Forecaster(ABC):
    """
    Abstract base class for all forecasters.

    Subclassing forces a consistent interface and gives static type-checkers
    something to verify. The actual scikit-learn protocol uses duck typing,
    but for our portfolio code an explicit ABC documents intent better.
    """

    name: str = "abstract"

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "Forecaster":
        """Learn from training data. Returns self for chaining."""

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Predict capacity factor for every row in df.index."""


# ---------------------------------------------------------------------------
# 1. Persistence
# ---------------------------------------------------------------------------

class Persistence(Forecaster):
    """
    Forecast: next hour equals this hour.

    The absolute floor — if a model can't beat this, it has learned nothing.
    No fitting required; predict from the input's own lagged values.

    Implementation note: this baseline needs the *target column* in the
    prediction-time dataframe, because it predicts y(t) = y(t-1). That seems
    like cheating, but it isn't — in operations, the previous hour's actual
    generation is known by the time you forecast the next hour.
    """

    name = "persistence"

    def __init__(self, target_col: str = "DE_solar_cf"):
        self.target_col = target_col

    def fit(self, df: pd.DataFrame) -> "Persistence":
        # Nothing to learn — this baseline is parameter-free.
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        # Predict t from t-1. The first row of the result is NaN (no t-1 exists),
        # which is handled correctly by evaluate() — it drops NaN-containing rows.
        return df[self.target_col].shift(1).rename("pred")


# ---------------------------------------------------------------------------
# 2. Daily seasonal naive
# ---------------------------------------------------------------------------

class DailySeasonalNaive(Forecaster):
    """
    Forecast: today at hour h equals yesterday at hour h.

    In solar forecasting this is a *surprisingly hard baseline to beat*. The
    sun rises and sets at nearly the same time on consecutive days, so the
    diurnal cycle is captured perfectly by lag-24. The only thing this
    baseline can't anticipate is weather changes between yesterday and today.
    """

    name = "daily_seasonal_naive"

    def __init__(self, target_col: str = "DE_solar_cf"):
        self.target_col = target_col

    def fit(self, df: pd.DataFrame) -> "DailySeasonalNaive":
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        return df[self.target_col].shift(24).rename("pred")


# ---------------------------------------------------------------------------
# 3. Hour-of-year climatology
# ---------------------------------------------------------------------------

class HourOfYearClimatology(Forecaster):
    """
    Forecast: capacity factor for (month, day, hour) equals the historical
    mean over training years.

    Pure seasonality, no awareness of recent conditions. Useful as a "what
    can be learned from the calendar alone" reference.

    We use (month, day, hour) rather than day-of-year because day-of-year shifts
    by one on leap years — using (month, day) keeps Feb 28 and Feb 29 aligned
    across years.
    """

    name = "hour_of_year_climatology"

    def __init__(self, target_col: str = "DE_solar_cf"):
        self.target_col = target_col
        self._table: pd.Series | None = None

    def fit(self, df: pd.DataFrame) -> "HourOfYearClimatology":
        idx = df.index
        keys = pd.MultiIndex.from_arrays(
            [idx.month, idx.day, idx.hour],
            names=["month", "day", "hour"],
        )
        self._table = (
            df[self.target_col]
              .groupby(keys)
              .mean()
              .rename("pred")
        )
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self._table is None:
            raise RuntimeError(f"{self.name} must be .fit() before .predict()")
        idx = df.index
        keys = pd.MultiIndex.from_arrays(
            [idx.month, idx.day, idx.hour],
            names=["month", "day", "hour"],
        )
        # Reindex the lookup table to the prediction timestamps. Any (month,
        # day, hour) tuple absent from training (e.g. Feb 29 if training had
        # no leap year) becomes NaN — honestly missing, not silently zero.
        out = self._table.reindex(keys)
        out.index = df.index
        return out


# ---------------------------------------------------------------------------
# 4. Linear regression on weather
# ---------------------------------------------------------------------------

class LinearWeather(Forecaster):
    """
    Forecast: y_hat = a·GHI + b·temperature + c.

    The simplest model that uses weather features. It's intentionally
    underpowered — no nonlinearity, no interactions, no seasonality, no
    lagged terms. We want the contrast: how much does *any* weather signal
    help, versus the calendar-only climatology baseline?

    Implementation note: we fit only on hours where both target and features
    are non-NaN, so the model isn't biased by missing data.
    """

    name = "linear_weather"

    def __init__(
        self,
        target_col: str = "DE_solar_cf",
        feature_cols: tuple[str, ...] = ("DE_ghi", "DE_temperature"),
    ):
        self.target_col = target_col
        self.feature_cols = list(feature_cols)
        self._coef: np.ndarray | None = None
        self._intercept: float | None = None

    def fit(self, df: pd.DataFrame) -> "LinearWeather":
        cols = [self.target_col, *self.feature_cols]
        clean = df[cols].dropna()
        X = clean[self.feature_cols].to_numpy()
        y = clean[self.target_col].to_numpy()

        # Least-squares via the normal equations with an intercept column.
        # We do this by hand (rather than calling sklearn) to keep the
        # baselines dependency-free — anyone reading the code sees exactly
        # what's happening. This is the only baseline where I'd consider that
        # tradeoff worth making; for anything more complex, use the library.
        X_with_intercept = np.hstack([X, np.ones((len(X), 1))])
        beta, *_ = np.linalg.lstsq(X_with_intercept, y, rcond=None)
        self._coef = beta[:-1]
        self._intercept = float(beta[-1])
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self._coef is None:
            raise RuntimeError(f"{self.name} must be .fit() before .predict()")
        X = df[self.feature_cols].to_numpy()
        y_hat = X @ self._coef + self._intercept
        # Clip to the physically valid range. The linear model occasionally
        # predicts slightly negative capacity factors on dark winter mornings,
        # which is meaningless — clip explicitly so downstream comparisons
        # aren't penalized by clearly-wrong outputs.
        y_hat = np.clip(y_hat, 0.0, 1.0)
        return pd.Series(y_hat, index=df.index, name="pred")


# ---------------------------------------------------------------------------
# 5. Linear regression on physics features
# ---------------------------------------------------------------------------

class LinearPhysics(LinearWeather):
    """
    Forecast: linear regression on physics-engineered features alongside raw weather.

    Subclasses LinearWeather and just overrides the default feature list —
    same fitting code, same prediction code, different inputs. This is what
    the uniform Forecaster interface is for.

    Design note (worth a paragraph in the eventual blog post):

    A naive instinct is to *replace* raw GHI and temperature with their
    physics-derived counterparts G_POA and T_cell. Empirically, that's
    *worse* for fleet-aggregated forecasting, because our G_POA assumes a
    single panel orientation while the real fleet has every tilt and azimuth.
    Raw GHI is a more honest measurement when the receiving geometry is
    heterogeneous.

    The right approach is to *augment*: keep the raw features, add the
    physics features, let the linear regression learn its own coefficients
    on each. This gives the model both the unbiased measurement (GHI) and
    the physics-shaped features that already encode the diurnal/temperature
    structure.
    """

    name = "linear_physics"

    def __init__(
        self,
        target_col: str = "DE_solar_cf",
        feature_cols: tuple[str, ...] = (
            # Raw weather (kept for fleet-heterogeneity reasons explained above)
            "DE_ghi",
            "DE_temperature",
            # Physics-engineered features
            "DE_g_poa",
            "DE_cf_theory",
            "DE_clearness_index",
            "DE_t_cell_c",
            "DE_solar_elevation_deg",
        ),
    ):
        super().__init__(target_col=target_col, feature_cols=feature_cols)


# ---------------------------------------------------------------------------
# 6. Pure-physics prediction (no fitting at all)
# ---------------------------------------------------------------------------

class PurePhysics(Forecaster):
    """
    Forecast: y_hat = cf_theory. Zero learned parameters.

    This is the most aggressive baseline philosophically — we're asking:
    "what if we *only* used physics, with no calibration to actual data?"

    If `cf_theory` is well-calibrated for the German fleet, this will be
    competitive with linear_physics. If it's not (likely it isn't — our
    site config is a guess at the fleet average), this will underperform,
    and that gap tells us how much "physics calibration" the linear models
    are doing.
    """

    name = "pure_physics"

    def __init__(self, theory_col: str = "DE_cf_theory"):
        self.theory_col = theory_col

    def fit(self, df: pd.DataFrame) -> "PurePhysics":
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        return df[self.theory_col].rename("pred").clip(0.0, 1.0)


# ---------------------------------------------------------------------------
# 7. XGBoost — the strong ML reference
# ---------------------------------------------------------------------------

class XGBoostForecaster(Forecaster):
    """
    Gradient-boosted trees on tabular features, same fit/predict interface
    as the rest. This is the "strong ML baseline" — if our PINN can't beat
    XGBoost on the same features, the physics-informed loss isn't earning
    its keep.

    Design choices worth noting:

    - **Early stopping uses an internal hold-out**, not the official val set.
      We carve the last 10% of training data (chronological, not random)
      for the early-stopping eval set. This keeps the official val set
      genuinely unseen by the model during fitting — otherwise we'd be
      tuning n_estimators against our reporting set, which is a subtle
      form of leakage.

    - **Hyperparameters are sensible defaults, not tuned.** This is
      deliberate. The story we want to tell is "XGBoost out of the box,
      no special effort." If we tuned, we'd be claiming a result that
      readers can't easily reproduce, and the comparison to the PINN
      would have to be against a tuned XGBoost too.

    - **XGBoost handles NaN natively** in features. We pass features
      through directly without filling. The first 168 hours of training
      data will have NaN lag-168 values; XGBoost learns a default
      direction for missing values at each split.
    """

    name = "xgboost"

    def __init__(
        self,
        target_col: str = "DE_solar_cf",
        feature_cols: tuple[str, ...] | None = None,
        n_estimators: int = 1000,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        early_stopping_rounds: int = 50,
        random_state: int = 42,
        early_stop_holdout_frac: float = 0.1,
    ):
        self.target_col = target_col
        self.feature_cols = list(feature_cols) if feature_cols is not None else None
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.early_stop_holdout_frac = early_stop_holdout_frac
        self._model = None
        self._fitted_features: list[str] | None = None

    def fit(self, df: pd.DataFrame) -> "XGBoostForecaster":
        # Imported lazily so the module doesn't require xgboost unless used —
        # keeps lightweight baseline runs from paying the import cost.
        import xgboost as xgb

        if self.feature_cols is None:
            raise ValueError(
                "XGBoostForecaster needs explicit feature_cols. Pass them in the constructor."
            )

        # Drop rows where the target is NaN; keep rows with NaN in features
        # (XGBoost handles them natively, and dropping would discard the
        # early-lag rows we want the model to learn what to do with).
        sorted_df = df.sort_index()
        usable = sorted_df.dropna(subset=[self.target_col])

        # Chronological hold-out for early stopping. We use the *last* slice
        # rather than a random sample so the validation distribution
        # resembles the future we want to predict.
        n_holdout = max(1, int(len(usable) * self.early_stop_holdout_frac))
        train_part = usable.iloc[:-n_holdout]
        eval_part = usable.iloc[-n_holdout:]

        X_train = train_part[self.feature_cols].to_numpy()
        y_train = train_part[self.target_col].to_numpy()
        X_eval = eval_part[self.feature_cols].to_numpy()
        y_eval = eval_part[self.target_col].to_numpy()

        self._model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            early_stopping_rounds=self.early_stopping_rounds,
            eval_metric="rmse",
            tree_method="hist",
            n_jobs=-1,
        )
        self._model.fit(
            X_train, y_train,
            eval_set=[(X_eval, y_eval)],
            verbose=False,
        )
        self._fitted_features = list(self.feature_cols)
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self._model is None:
            raise RuntimeError(f"{self.name} must be .fit() before .predict()")
        X = df[self._fitted_features].to_numpy()
        y_hat = self._model.predict(X)
        # Capacity factor is bounded; clip to ensure no nonsense values
        # in the rare cases XGBoost extrapolates past the training envelope.
        import numpy as np
        y_hat = np.clip(y_hat, 0.0, 1.0)
        return pd.Series(y_hat, index=df.index, name="pred")

    @property
    def feature_importance(self) -> pd.Series | None:
        """Gain-based feature importance from the trained model."""
        if self._model is None or self._fitted_features is None:
            return None
        return pd.Series(
            self._model.feature_importances_,
            index=self._fitted_features,
            name="importance",
        ).sort_values(ascending=False)
