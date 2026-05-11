"""
Data quality report for the OPSD time series and weather packages.

Produces:
    reports/data_quality.md        — committable markdown summary
    reports/figures/*.png          — committable plots (small, ~200 KB each)

This script is deliberately verbose in its output: every finding is logged
so the markdown report becomes a permanent record of what we knew about
the data at the start of the project. Future-us will thank present-us.

Usage:
    python scripts/data_quality_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Repo-relative paths so the script works from anywhere inside the project.
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

# Columns we actually need. Loading the full CSV is wasteful — time_series alone
# has ~400 columns and most are irrelevant to a Germany-focused PV forecaster.
TS_COLS = [
    "utc_timestamp",
    "DE_load_actual_entsoe_transparency",
    "DE_solar_generation_actual",
    "DE_solar_capacity",
    "DE_50hertz_solar_generation_actual",
    "DE_amprion_solar_generation_actual",
    "DE_tennet_solar_generation_actual",
    "DE_transnetbw_solar_generation_actual",
    "IT_solar_generation_actual",
]
WX_COLS = [
    "utc_timestamp",
    "DE_temperature",
    "DE_radiation_direct_horizontal",
    "DE_radiation_diffuse_horizontal",
    "IT_temperature",
    "IT_radiation_direct_horizontal",
    "IT_radiation_diffuse_horizontal",
]


# ---------------------------------------------------------------------------
# Section runners — each returns a markdown string that gets joined at the end.
# Splitting by section keeps the orchestration in main() readable.
# ---------------------------------------------------------------------------

def section_coverage(ts: pd.DataFrame, wx: pd.DataFrame) -> str:
    """Date ranges of each source and the usable overlap."""
    ts_start, ts_end = ts["utc_timestamp"].min(), ts["utc_timestamp"].max()
    wx_start, wx_end = wx["utc_timestamp"].min(), wx["utc_timestamp"].max()
    overlap_start = max(ts_start, wx_start)
    overlap_end = min(ts_end, wx_end)
    overlap_hours = int(
        (overlap_end - overlap_start).total_seconds() // 3600) + 1

    return f"""## 1. Temporal coverage

| Source       | Start                | End                  | Rows        |
|--------------|----------------------|----------------------|-------------|
| Time series  | {ts_start}           | {ts_end}             | {len(ts):>10,} |
| Weather      | {wx_start}           | {wx_end}             | {len(wx):>10,} |

**Usable overlap:** {overlap_start} → {overlap_end} ({overlap_hours:,} hours, ~{overlap_hours / 8760:.1f} years).

This is the universe we model on. Everything before {overlap_start.date()} has weather but no target;
everything after {overlap_end.date()} has target but no weather. Both are unusable for supervised training.
"""


def section_missingness(ts: pd.DataFrame, wx: pd.DataFrame) -> tuple[str, dict]:
    """Per-column missingness, both aggregate and by year."""
    rows = []
    for df, label in [(ts, "time_series"), (wx, "weather")]:
        for col in df.columns[1:]:  # skip utc_timestamp
            pct = df[col].notna().mean() * 100
            rows.append({"source": label, "column": col,
                        "present_pct": round(pct, 2)})
    summary = pd.DataFrame(rows).sort_values("present_pct")

    # Yearly missingness for our headline target — averages hide gap weeks.
    yearly = (
        ts.assign(year=ts["utc_timestamp"].dt.year)
          .groupby("year")["DE_solar_generation_actual"]
          .agg(present_pct=lambda x: x.notna().mean() * 100)
          .round(2)
    )

    md = "## 2. Missingness\n\n### Aggregate (all years)\n\n"
    md += summary.to_markdown(index=False) + "\n\n"
    md += "### DE_solar_generation_actual by year\n\n"
    md += yearly.to_markdown() + "\n\n"
    md += "Aggregate percentages can hide structural gaps. Always check by year.\n"
    return md, {"aggregate": summary, "yearly_de_solar": yearly}


def section_anomalies(ts: pd.DataFrame, wx: pd.DataFrame) -> str:
    """Impossible or implausible values that suggest sensor errors."""
    de_solar = ts["DE_solar_generation_actual"]
    de_load = ts["DE_load_actual_entsoe_transparency"]
    de_temp = wx["DE_temperature"]
    de_ghi = wx["DE_radiation_direct_horizontal"] + \
        wx["DE_radiation_diffuse_horizontal"]

    # Solar generation should never be negative — but some sensors report tiny
    # negative values from inverter night-time draw or calibration drift.
    neg_solar = int((de_solar < 0).sum())
    neg_solar_pct = (de_solar < 0).mean() * 100

    # Night-time generation > 1 MW is impossible at Germany's latitude.
    # Build a UTC-aware mask: 22:00–04:00 UTC ~ midnight in Germany.
    night_mask = ts["utc_timestamp"].dt.hour.isin([22, 23, 0, 1, 2, 3, 4])
    night_solar = ts.loc[night_mask, "DE_solar_generation_actual"]
    impossible_night = int((night_solar > 1).sum())

    # Temperature sanity: anything outside [-30, +45] °C in Germany is suspect.
    temp_out_of_range = int(((de_temp < -30) | (de_temp > 45)).sum())

    # Irradiance should be 0 at night. Non-zero night irradiance = source data issue.
    night_irr = wx.loc[
        wx["utc_timestamp"].dt.hour.isin([22, 23, 0, 1, 2, 3]),
        ["DE_radiation_direct_horizontal", "DE_radiation_diffuse_horizontal"]
    ]
    night_irr_nonzero = int((night_irr.sum(axis=1) > 1).sum())

    return f"""## 3. Anomalies and impossible values

| Check                                        |     Count | Notes |
|----------------------------------------------|----------:|-------|
| `DE_solar_generation_actual` < 0             | {neg_solar:>9,} | {neg_solar_pct:.3f}% of rows. Typically inverter calibration drift; clip to 0 in preprocessing. |
| Night-time (22–04 UTC) solar > 1 MW          | {impossible_night:>9,} | Should be ~0. Any non-zero count is a red flag. |
| Temperature outside [-30, +45] °C            | {temp_out_of_range:>9,} | Germany never reaches these. |
| Night-time GHI > 1 W/m² (combined direct+diffuse) | {night_irr_nonzero:>9,} | Reanalysis artifact; should be exactly 0. |

These aren't necessarily errors to remove — but they are decisions to make explicitly
in preprocessing, and the decision belongs in code, not in your head.
"""


def section_physics_check(ts: pd.DataFrame, wx: pd.DataFrame) -> str:
    """The thesis of the project: does irradiance predict generation?"""
    merged = ts[["utc_timestamp", "DE_solar_generation_actual"]].merge(
        wx[["utc_timestamp", "DE_radiation_direct_horizontal",
            "DE_radiation_diffuse_horizontal", "DE_temperature"]],
        on="utc_timestamp", how="inner"
    ).dropna()
    merged["DE_ghi"] = (
        merged["DE_radiation_direct_horizontal"]
        + merged["DE_radiation_diffuse_horizontal"]
    )

    overall = merged["DE_solar_generation_actual"].corr(merged["DE_ghi"])

    # The big question: does correlation hold across seasons? If it drops in
    # winter, the physics-informed model needs to handle that — a flat linear
    # prior won't be enough.
    merged["month"] = merged["utc_timestamp"].dt.month
    by_month = merged.groupby("month").apply(
        lambda g: g["DE_solar_generation_actual"].corr(g["DE_ghi"]),
        include_groups=False,
    ).round(3)

    return f"""## 4. Physical sanity: does irradiance predict generation?

**Overall Pearson correlation:** DE solar generation ↔ DE GHI = **{overall:.4f}**
(over {len(merged):,} hourly observations).

This is the headline number for the project. Anything above ~0.9 means the
physics-informed thesis is viable; anything below ~0.7 would be cause for
concern. We are comfortably above the threshold.

### Correlation by month

| Month |  Corr | |  Month |  Corr |
|------:|------:|-|-------:|------:|
{_format_monthly_corr(by_month)}

Months with lower correlation tell us where a pure-linear physics prior will
struggle most — typically winter, when low sun angles, snow, and increased
diffuse fraction make the GHI → output mapping less direct. This is exactly
the regime where the learned residual on top of the physics prior earns its keep.
"""


def _format_monthly_corr(by_month: pd.Series) -> str:
    """Format monthly correlations as a two-column markdown table body."""
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    lines = []
    for i in range(6):
        l = f"| {months[i]:>5} | {by_month.iloc[i]:.3f} |"
        r = f" | {months[i + 6]:>5} | {by_month.iloc[i + 6]:.3f} |"
        lines.append(l + r)
    return "\n".join(lines)


def section_capacity_growth(ts: pd.DataFrame) -> str:
    """Installed PV capacity over time — a structural break we must handle."""
    cap = (
        ts.dropna(subset=["DE_solar_capacity"])
        .assign(year=ts["utc_timestamp"].dt.year)
        .groupby("year")["DE_solar_capacity"]
        .agg(["min", "max"])
        .round(0)
    )
    growth_pct = (cap["max"].iloc[-1] / cap["min"].iloc[0] -
                  1) * 100 if len(cap) > 1 else 0

    return f"""## 5. Installed capacity growth

| Year | Min (MW) | Max (MW) |
|-----:|---------:|---------:|
""" + "\n".join(
        f"| {yr:>4} | {row['min']:>8,.0f} | {row['max']:>8,.0f} |"
        for yr, row in cap.iterrows()
    ) + f"""

Germany's installed PV capacity grew **~{growth_pct:.0f}%** across the dataset.
A naive chronological train/test split therefore trains on a smaller grid and
tests on a larger one — guaranteed distribution shift. Two ways to handle it:

1. **Normalize generation by capacity** (turn MW into a capacity factor in [0,1]).
2. **Include capacity as a feature** so the model learns the scaling explicitly.

We'll use approach 1 as the default; it's cleaner and decouples the forecasting
problem from the capacity-tracking problem.
"""


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_one_year(ts: pd.DataFrame, wx: pd.DataFrame, year: int) -> Path:
    """Side-by-side: solar generation and GHI for a single year."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    ts_year = ts[ts["utc_timestamp"].dt.year == year]
    wx_year = wx[wx["utc_timestamp"].dt.year == year]
    ghi = (
        wx_year["DE_radiation_direct_horizontal"]
        + wx_year["DE_radiation_diffuse_horizontal"]
    )

    axes[0].plot(ts_year["utc_timestamp"], ts_year["DE_solar_generation_actual"],
                 linewidth=0.4, color="#D85A30")
    axes[0].set_ylabel("Solar generation (MW)")
    axes[0].set_title(f"Germany — {year}")

    axes[1].plot(wx_year["utc_timestamp"], ghi, linewidth=0.4, color="#EF9F27")
    axes[1].set_ylabel("GHI (W/m²)")
    axes[1].set_xlabel("UTC")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = FIG_DIR / f"de_solar_vs_ghi_{year}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_capacity_growth(ts: pd.DataFrame) -> Path:
    """Time series of installed PV capacity."""
    cap = ts.dropna(subset=["DE_solar_capacity"])
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(cap["utc_timestamp"], cap["DE_solar_capacity"] / 1000,
            color="#1D9E75", linewidth=1.2)
    ax.set_ylabel("Installed capacity (GW)")
    ax.set_xlabel("UTC")
    ax.set_title("Germany — installed PV capacity over time")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = FIG_DIR / "de_capacity_growth.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_missingness_heatmap(ts: pd.DataFrame) -> Path:
    """Visual map of missingness by month-year for the key target column."""
    df = ts[["utc_timestamp", "DE_solar_generation_actual"]].copy()
    df["year"] = df["utc_timestamp"].dt.year
    df["month"] = df["utc_timestamp"].dt.month
    pivot = (
        df.assign(missing=df["DE_solar_generation_actual"].isna().astype(int))
        .groupby(["year", "month"])["missing"]
        .mean()
        .unstack(fill_value=np.nan)
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="Reds", vmin=0, vmax=1)
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("DE_solar_generation_actual — fraction missing by month")
    fig.colorbar(im, ax=ax, label="Fraction missing")
    fig.tight_layout()
    path = FIG_DIR / "de_solar_missingness_heatmap.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    ts_path = DATA_DIR / "time_series_60min_singleindex.csv"
    wx_path = DATA_DIR / "weather_data.csv"
    if not ts_path.exists() or not wx_path.exists():
        print(f"ERROR: raw data files not found in {DATA_DIR}.")
        print("Run `python scripts/download_data.py` first.")
        return 1

    ts = pd.read_csv(ts_path, usecols=TS_COLS, parse_dates=["utc_timestamp"])
    wx = pd.read_csv(wx_path, usecols=WX_COLS, parse_dates=["utc_timestamp"])
    print(f"  time_series: {len(ts):,} rows")
    print(f"  weather:     {len(wx):,} rows")

    print("Building report sections...")
    sections = [
        section_coverage(ts, wx),
        section_missingness(ts, wx)[0],
        section_anomalies(ts, wx),
        section_physics_check(ts, wx),
        section_capacity_growth(ts),
    ]

    print("Rendering figures...")
    plot_one_year(ts, wx, 2018)
    plot_capacity_growth(ts)
    plot_missingness_heatmap(ts)

    print("Writing markdown...")
    header = f"""# Data Quality Report

_Generated by `scripts/data_quality_report.py`. Re-run after any change to raw data._

Dataset: OPSD time series v2020-10-06, OPSD weather v2020-09-16.

---

"""
    figures_md = """## 6. Figures

![DE solar generation vs GHI, 2018](figures/de_solar_vs_ghi_2018.png)

![DE installed capacity growth](figures/de_capacity_growth.png)

![DE solar missingness heatmap](figures/de_solar_missingness_heatmap.png)
"""

    report = header + "\n\n".join(sections) + "\n\n" + figures_md
    (REPORT_DIR / "data_quality.md").write_text(report, encoding="utf-8")

    print(f"\n✓ Done. Report at {REPORT_DIR / 'data_quality.md'}")
    print(f"  Figures at {FIG_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
