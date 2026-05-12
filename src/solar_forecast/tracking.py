"""
MLflow experiment tracking.

Backend is SQLite (a single file, `mlflow.db` in the repo root). Reasons:

1. **Cross-platform.** The legacy file:// backend doesn't accept Windows
   paths like file://C:\\Users\\... — MLflow's URI validator rejects them.
   SQLite uses a sqlite:// URI that works identically on every OS.
2. **MLflow recommends it.** As of Feb 2026 the file-based backend is
   deprecated in favor of database-backed stores.
3. **Single file is easier to gitignore and inspect.**

Open the UI from the repo root with:
    mlflow ui --backend-store-uri sqlite:///mlflow.db

Or set the env var once per shell:
    export MLFLOW_TRACKING_URI=sqlite:///mlflow.db   # bash/zsh
    $env:MLFLOW_TRACKING_URI = "sqlite:///mlflow.db" # PowerShell
then just `mlflow ui` works.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "mlflow.db"
ARTIFACTS_DIR = ROOT / "mlruns_artifacts"  # for things like CSVs we log()

# sqlite:/// + absolute path. Convert the Path to a forward-slash string so
# the URI is identical on Windows and Unix.
TRACKING_URI = f"sqlite:///{DB_PATH.as_posix()}"


def configure(experiment_name: str = "solar-forecast-baselines") -> None:
    """Set the MLflow tracking URI and experiment. Idempotent."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment_name)


def log_model_run(
    *,
    model_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    feature_cols: list[str] | None = None,
    feature_importance: pd.Series | None = None,
) -> None:
    """
    Log a single model's results as an MLflow run.

    Keeps the runner script clean — one call per model.
    """
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params({k: str(v) for k, v in params.items()})
        if feature_cols is not None:
            mlflow.log_param("n_features", len(feature_cols))
            mlflow.log_param("features", ",".join(feature_cols))
        mlflow.log_metrics(metrics)
        if feature_importance is not None:
            tmp = ARTIFACTS_DIR / f"_tmp_importance_{model_name}.csv"
            feature_importance.to_csv(tmp, header=True)
            mlflow.log_artifact(str(tmp))
            tmp.unlink(missing_ok=True)
