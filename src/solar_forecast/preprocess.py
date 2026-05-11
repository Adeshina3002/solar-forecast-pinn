"""
Preprocessing pipeline: raw OPSD CSVs → clean modeling-ready parquet.

Each preprocessing decision identified in `reports/data_quality.md` is its own
function here, with a docstring explaining the why and a unit test in
`tests/test_preprocess.py`. The orchestrator at the bottom is deliberately dumb
glue — all the logic lives in the small functions, which keeps them testable
and lets you reason about one decision at a time.

The cached output is parquet, not CSV. Parquet is ~10x faster to load and
preserves dtypes (notably tz-aware timestamps), which CSV does not.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Repo-relative defaults. These are overrideable by passing explicit paths to
# build_dataset() — important for tests, which run on synthetic fixtures.
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = ROOT / "data" / "processed"

# Columns we load. Listing them explicitly (rather than read_csv(usecols=None))
# is a contract: the pipeline depends on these columns existing. If OPSD ever
# renames one, the failure happens at load time with a clear error, not deep
# inside a model training loop.
TS_COLUMNS = [
    "utc_timestamp",
    "DE_load_actual_entsoe_transparency",
    "DE_solar_generation_actual",
    "DE_solar_capacity",
    "IT_solar_generation_actual",
]
WX_COLUMNS = [
    "utc_timestamp",
    "DE_temperature",
    "DE_radiation_direct_horizontal",
    "DE_radiation_diffuse_horizontal",
    "IT_temperature",
    "IT_radiation_direct_horizontal",
    "IT_radiation_diffuse_horizontal",
]

# Hours we treat as "night" in UTC. Germany is UTC+1 (winter) / UTC+2 (summer),
# so 22:00-04:00 UTC reliably falls between sunset and sunrise year-round.
# This is conservative — we'd rather miss a few twilight hours than mis-clip a
# real generation hour.
NIGHT_HOURS_UTC = {22, 23, 0, 1, 2, 3, 4}


@dataclass(frozen=True)
class PreprocessConfig:
    """
    Tunable knobs of the pipeline, gathered in one place.

    Frozen so they can't be mutated mid-run (a class of bug that's miserable to
    debug). Override by constructing a new instance, not by mutation.
    """
    overlap_start: str = "2015-01-01"
    overlap_end: str = "2019-12-31 23:00:00"
    night_solar_threshold_mw: float = 1.0     # below this we clip to zero
    night_ghi_threshold_w_m2: float = 1.0     # below this we clip to zero
    max_interpolation_gap_hours: int = 3      # gaps longer than this stay NaN


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_raw(
    raw_dir: Path = DEFAULT_RAW_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the two OPSD CSVs with only the columns we use.

    Returns (time_series, weather), both with utc_timestamp parsed to
    tz-aware datetime.
    """
    ts_path = raw_dir / "time_series_60min_singleindex.csv"
    wx_path = raw_dir / "weather_data.csv"

    for path in (ts_path, wx_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Raw data not found at {path}. "
                f"Run `python scripts/download_data.py` first."
            )

    ts = pd.read_csv(ts_path, usecols=TS_COLUMNS, parse_dates=["utc_timestamp"])
    wx = pd.read_csv(wx_path, usecols=WX_COLUMNS, parse_dates=["utc_timestamp"])
    logger.info("Loaded %d time-series rows, %d weather rows", len(ts), len(wx))
    return ts, wx


# ---------------------------------------------------------------------------
# Cleaning steps — one decision per function
# ---------------------------------------------------------------------------

def clip_nighttime_solar(
    df: pd.DataFrame,
    *,
    solar_col: str,
    threshold_mw: float,
) -> pd.DataFrame:
    """
    Force nighttime solar generation to zero.

    The data quality report flagged 1,952 hours of >1 MW solar generation
    between 22:00 and 04:00 UTC. At Germany's latitude this is physically
    impossible (sun is below the horizon). These come from TSO aggregation
    artifacts where small biomass/conventional inverters get rolled into
    "solar" by some reporting feeds.

    We don't drop these rows — that would leave holes in the time series.
    We clip the value to zero.
    """
    df = df.copy()
    night = df["utc_timestamp"].dt.hour.isin(NIGHT_HOURS_UTC)
    suspicious = night & (df[solar_col] > threshold_mw)
    n_fixed = int(suspicious.sum())
    df.loc[suspicious, solar_col] = 0.0
    logger.info("Clipped %d nighttime %s hours to zero", n_fixed, solar_col)
    return df


def clip_nighttime_irradiance(
    df: pd.DataFrame,
    *,
    direct_col: str,
    diffuse_col: str,
    threshold_w_m2: float,
) -> pd.DataFrame:
    """
    Force nighttime irradiance to zero.

    MERRA-2 reanalysis has small numerical residuals in its radiative transfer
    model — the data quality report found 3,523 nighttime hours with non-zero
    GHI. These should be exactly zero (sun below horizon). We clip both
    direct and diffuse components.
    """
    df = df.copy()
    night = df["utc_timestamp"].dt.hour.isin(NIGHT_HOURS_UTC)
    ghi = df[direct_col] + df[diffuse_col]
    suspicious = night & (ghi > threshold_w_m2)
    n_fixed = int(suspicious.sum())
    df.loc[suspicious, [direct_col, diffuse_col]] = 0.0
    logger.info("Clipped %d nighttime irradiance hours to zero", n_fixed)
    return df


def normalize_to_capacity_factor(
    df: pd.DataFrame,
    *,
    generation_col: str,
    capacity_col: str,
    output_col: str,
) -> pd.DataFrame:
    """
    Convert raw MW generation into capacity factor in [0, 1].

    Germany's installed PV capacity grew 36% over our 5-year window. Without
    this normalization, the model has to learn both "how does weather drive
    output" and "how big is the grid right now" simultaneously. Normalizing
    separates the two: we forecast capacity factor (a stationary quantity),
    and capacity scaling is applied separately at inference time.

    Edge cases:
    - capacity_col missing (NaN) → output is NaN (we drop these rows later)
    - generation_col > capacity_col → clipped to 1.0 (rare; rounding artifact)
    - generation_col < 0 → clipped to 0.0 (rare; sensor calibration drift)
    """
    df = df.copy()
    cf = df[generation_col] / df[capacity_col]
    cf = cf.clip(lower=0.0, upper=1.0)
    df[output_col] = cf
    n_clipped_high = int((df[generation_col] > df[capacity_col]).sum())
    if n_clipped_high:
        logger.info("Clipped %d hours where generation > capacity", n_clipped_high)
    return df


def align_to_window(
    df: pd.DataFrame,
    *,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Subset to the modeling window and re-index to a strict hourly grid.

    Re-indexing to a regular hourly DatetimeIndex catches two classes of bug:
    (a) duplicate timestamps in the source (we'd silently average them),
    (b) missing hours (NaN-filled rather than silently absent).

    Either case becomes visible NaN counts downstream — exactly what we want.
    """
    df = df.copy()
    df = df.sort_values("utc_timestamp")

    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    mask = (df["utc_timestamp"] >= start_ts) & (df["utc_timestamp"] <= end_ts)
    df = df.loc[mask].set_index("utc_timestamp")

    full_index = pd.date_range(start_ts, end_ts, freq="h", tz="UTC", name="utc_timestamp")
    return df.reindex(full_index)


def interpolate_short_gaps(
    df: pd.DataFrame,
    *,
    columns: list[str],
    max_gap_hours: int,
) -> pd.DataFrame:
    """
    Fill gaps shorter than `max_gap_hours` by linear interpolation.

    Long gaps stay as NaN — we'd rather have honest missing data than fabricated
    weather. The downstream training code drops rows with any NaN in the
    feature/target set.

    Note: pandas' built-in `interpolate(limit=n)` fills *up to* n NaNs from each
    side of valid data, which would leave us fabricating the first n hours of
    every long gap. We explicitly measure run lengths and only fill runs
    entirely within the threshold.
    """
    df = df.copy()
    for col in columns:
        series = df[col]
        is_nan = series.isna()

        # Assign each contiguous NaN run a unique group id. The trick: cumsum
        # of "not NaN" increments on every non-NaN value, so all NaNs between
        # two non-NaNs share the same id.
        run_id = (~is_nan).cumsum()
        run_lengths = is_nan.groupby(run_id).transform("sum")
        is_short_gap = is_nan & (run_lengths <= max_gap_hours)

        interpolated = series.interpolate(method="time", limit_area="inside")
        df[col] = series.where(~is_short_gap, interpolated)

        filled = int(is_short_gap.sum())
        if filled > 0:
            logger.info("Filled %d short-gap hours in %s by interpolation", filled, col)
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features the model will need that derive cheaply from raw columns.

    Right now: just total GHI (direct + diffuse) for both countries. More
    features will land here as the project progresses (sun position, NOCT
    cell temperature, lagged generation, etc.) — but those each get their
    own ticket. Don't try to do all feature engineering here.
    """
    df = df.copy()
    df["DE_ghi"] = df["DE_radiation_direct_horizontal"] + df["DE_radiation_diffuse_horizontal"]
    df["IT_ghi"] = df["IT_radiation_direct_horizontal"] + df["IT_radiation_diffuse_horizontal"]
    return df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_dataset(
    raw_dir: Path = DEFAULT_RAW_DIR,
    config: PreprocessConfig | None = None,
) -> pd.DataFrame:
    """
    Run the full preprocessing pipeline and return the clean dataframe.

    This is the function downstream code (training, evaluation, notebooks)
    calls. Caching to parquet is handled by build_and_cache() below.
    """
    cfg = config or PreprocessConfig()
    ts_raw, wx_raw = load_raw(raw_dir)

    # Clean each source independently — this keeps the steps composable and
    # the failure modes narrow.
    ts = clip_nighttime_solar(
        ts_raw,
        solar_col="DE_solar_generation_actual",
        threshold_mw=cfg.night_solar_threshold_mw,
    )
    ts = clip_nighttime_solar(
        ts,
        solar_col="IT_solar_generation_actual",
        threshold_mw=cfg.night_solar_threshold_mw,
    )
    wx = clip_nighttime_irradiance(
        wx_raw,
        direct_col="DE_radiation_direct_horizontal",
        diffuse_col="DE_radiation_diffuse_horizontal",
        threshold_w_m2=cfg.night_ghi_threshold_w_m2,
    )
    wx = clip_nighttime_irradiance(
        wx,
        direct_col="IT_radiation_direct_horizontal",
        diffuse_col="IT_radiation_diffuse_horizontal",
        threshold_w_m2=cfg.night_ghi_threshold_w_m2,
    )

    # Capacity factor is built before alignment so the column exists in both
    # the merged and the unmerged dataframes — useful for diagnostics.
    ts = normalize_to_capacity_factor(
        ts,
        generation_col="DE_solar_generation_actual",
        capacity_col="DE_solar_capacity",
        output_col="DE_solar_cf",
    )

    # Align both to the same hourly grid, then merge.
    ts = align_to_window(ts, start=cfg.overlap_start, end=cfg.overlap_end)
    wx = align_to_window(wx, start=cfg.overlap_start, end=cfg.overlap_end)
    df = ts.join(wx, how="inner")

    df = interpolate_short_gaps(
        df,
        columns=[
            "DE_solar_generation_actual", "DE_solar_cf",
            "DE_load_actual_entsoe_transparency",
        ],
        max_gap_hours=cfg.max_interpolation_gap_hours,
    )

    df = add_derived_features(df)

    logger.info(
        "Final dataset: %d rows × %d cols, %.2f%% complete on DE_solar_cf",
        len(df), len(df.columns),
        df["DE_solar_cf"].notna().mean() * 100,
    )
    return df


def build_and_cache(
    raw_dir: Path = DEFAULT_RAW_DIR,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    config: PreprocessConfig | None = None,
    force: bool = False,
) -> Path:
    """
    Run the pipeline and cache the result as parquet.

    Idempotent: if the parquet already exists and force=False, just returns
    its path without rebuilding. This is what scripts and notebooks should
    call — building the dataset from raw CSV takes ~10 seconds, loading the
    parquet takes ~100 ms.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "dataset.parquet"

    if out_path.exists() and not force:
        logger.info("Cache hit at %s (use force=True to rebuild)", out_path)
        return out_path

    df = build_dataset(raw_dir=raw_dir, config=config)
    df.to_parquet(out_path, engine="pyarrow", compression="snappy")
    logger.info("Wrote %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)
    return out_path


def load_dataset(processed_dir: Path = DEFAULT_PROCESSED_DIR) -> pd.DataFrame:
    """
    Load the cached preprocessed dataset.

    The one-line function every notebook and training script should start with.
    Raises a clear error if the cache hasn't been built yet.
    """
    path = processed_dir / "dataset.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {path}. "
            f"Run `python scripts/preprocess.py` first."
        )
    return pd.read_parquet(path)
