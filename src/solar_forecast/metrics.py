"""
Evaluation metrics for solar forecasting.

Centralizing the metrics here serves the same purpose as centralizing the
splits: every model must report the same numbers, computed the same way,
or comparisons across models are meaningless.

We report three:
    RMSE — penalizes large errors; the standard headline metric
    MAE  — robust to outliers; closer to what operators care about
    skill_score — relative improvement over a reference baseline,
                  bounded in (-inf, 1]; 0 = same as baseline, 1 = perfect

Skill score is the metric that matters most for a portfolio project. Anyone
can compute RMSE. The skill score over persistence is what tells the reader
"my model is X% better than the dumbest possible thing."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Metrics:
    """Headline metrics for a single forecast run."""
    rmse: float
    mae: float
    n_samples: int

    def __str__(self) -> str:
        return f"RMSE={self.rmse:.4f}  MAE={self.mae:.4f}  n={self.n_samples:,}"


def evaluate(y_true: pd.Series, y_pred: pd.Series) -> Metrics:
    """
    Compute RMSE and MAE for aligned series.

    The series are aligned by index before computation — this is the only
    safe way to compare them, and it catches the common bug of off-by-one
    time alignment between predictions and targets.

    Rows with NaN in either series are dropped (a model that can't predict
    every hour shouldn't be punished for those hours, but should be punished
    via the n_samples count being lower).
    """
    aligned = pd.concat([y_true.rename("true"), y_pred.rename("pred")], axis=1).dropna()
    if aligned.empty:
        raise ValueError("No overlapping non-NaN values between y_true and y_pred")

    err = aligned["true"] - aligned["pred"]
    rmse = float(np.sqrt((err ** 2).mean()))
    mae = float(err.abs().mean())
    return Metrics(rmse=rmse, mae=mae, n_samples=len(aligned))


def skill_score(
    candidate: Metrics,
    baseline: Metrics,
    metric: str = "rmse",
) -> float:
    """
    Relative improvement of candidate over baseline.

    Definition (mirrors the meteorology convention):
        skill = 1 − (candidate_error / baseline_error)

    Interpretation:
        skill = 0   → candidate is identical to baseline
        skill = 1   → candidate is perfect (zero error)
        skill < 0  → candidate is *worse* than baseline (a real signal!)
    """
    if metric not in {"rmse", "mae"}:
        raise ValueError(f"metric must be 'rmse' or 'mae', got {metric!r}")
    cand_err = getattr(candidate, metric)
    base_err = getattr(baseline, metric)
    if base_err == 0:
        return float("nan")  # candidate either also perfect or much worse — degenerate
    return 1.0 - (cand_err / base_err)
