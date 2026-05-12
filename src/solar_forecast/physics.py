"""
Physics-based feature engineering for solar forecasting.

Every function here computes something *derivable from first principles* —
no learned parameters, no fitting. These are deterministic transforms of
location, time, and weather into quantities that closer reflect what a real
PV panel experiences.

Why this matters: a linear model on `G_POA` and `T_cell` is doing something
fundamentally different from a linear model on raw `GHI` and `T_ambient`.
The first has the geometry of the sun-panel relationship and the
temperature-derating physics baked in. The model only has to learn
the residual — the discrepancy between what physics predicts and what the
grid actually produced.

References used (and worth bookmarking):

- Duffie, Beckman. "Solar Engineering of Thermal Processes" (4th ed.).
  The canonical textbook for the equations below.
- Reno, Hansen, Stein (Sandia, 2012). "Global Horizontal Irradiance Clear Sky
  Models: Implementation and Analysis." Source for the simple Haurwitz
  clear-sky model.
- Skoplaki & Palyvos (2009). "On the temperature dependence of photovoltaic
  module electrical performance." Source for the NOCT cell temperature model
  and the typical temperature coefficient.

Units: SI throughout. Angles in degrees in the public API (converted to
radians internally where the math needs it). Temperature in degrees Celsius.
Irradiance in W/m².
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants and configuration
# ---------------------------------------------------------------------------

# Default site parameters for the German PV fleet — chosen as a reasonable
# nationwide average, since OPSD aggregates over the whole country. These
# numbers come from the BNetzA Marktstammdatenregister and SolarPower Europe
# fleet statistics: typical residential/commercial tilt around latitude, mostly
# south-facing. A future improvement is to do this per-TSO with different
# parameters per zone.
DEFAULT_LATITUDE_DEG = 51.0       # central Germany, weighted toward Bavaria/NRW
DEFAULT_LONGITUDE_DEG = 10.5
DEFAULT_TILT_DEG = 35.0           # typical pitched-roof angle in Germany
DEFAULT_AZIMUTH_DEG = 180.0       # south-facing (north = 0, east = 90, south = 180)

# PV system constants — module-level rather than instance-level because these
# values are roughly universal for crystalline silicon panels of the era.
NOCT_DEG_C = 45.0                 # Nominal Operating Cell Temperature
NOCT_REFERENCE_IRRADIANCE = 800.0 # W/m², the irradiance at which NOCT is defined
NOCT_REFERENCE_AIR_TEMP = 20.0    # °C, ambient temperature at NOCT conditions
TEMP_COEFFICIENT_PER_C = 0.004    # 0.4%/°C — power lost per degree above 25°C
STC_REFERENCE_TEMP = 25.0         # °C, Standard Test Conditions cell temperature
SOLAR_CONSTANT = 1361.0           # W/m², total solar irradiance outside the atmosphere


@dataclass(frozen=True)
class SiteConfig:
    """Physical parameters of the PV site we model."""
    latitude_deg: float = DEFAULT_LATITUDE_DEG
    longitude_deg: float = DEFAULT_LONGITUDE_DEG
    tilt_deg: float = DEFAULT_TILT_DEG
    azimuth_deg: float = DEFAULT_AZIMUTH_DEG


# ---------------------------------------------------------------------------
# Solar position
# ---------------------------------------------------------------------------

def solar_position(
    timestamps: pd.DatetimeIndex,
    latitude_deg: float,
    longitude_deg: float,
) -> pd.DataFrame:
    """
    Compute solar elevation and azimuth for each UTC timestamp.

    Returns a DataFrame with two columns:
        elevation_deg : angle of the sun above the horizon (0–90, negative at night)
        azimuth_deg   : compass direction of the sun (0 = north, 180 = south)
        zenith_deg    : 90 - elevation, convenient for downstream math

    The formulas here are the standard ones from Duffie & Beckman, Chapter 1.
    Accuracy is ±0.5° — plenty for solar forecasting where the weather data
    itself has much larger uncertainty.
    """
    if timestamps.tz is None:
        raise ValueError("timestamps must be tz-aware (UTC)")

    # Day of year and fractional hour in UTC
    day_of_year = timestamps.dayofyear.to_numpy()
    hour_utc = (
        timestamps.hour
        + timestamps.minute / 60.0
        + timestamps.second / 3600.0
    ).to_numpy()

    # Equation of time (minutes): correction for Earth's elliptical orbit and
    # axial tilt. The B angle formulation from Spencer 1971 is standard.
    B = np.deg2rad((day_of_year - 81) * 360.0 / 365.0)
    eq_of_time_min = 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)

    # Solar declination (degrees): tilt of Earth's axis relative to the
    # sun-Earth line on this day of year.
    declination_deg = 23.45 * np.sin(np.deg2rad((day_of_year - 81) * 360.0 / 365.0))

    # Local solar time (hours). Longitude shifts the solar noon: each 15° east
    # advances solar noon by 1 hour relative to UTC.
    solar_time_hr = hour_utc + longitude_deg / 15.0 + eq_of_time_min / 60.0

    # Hour angle (degrees): how far the sun is from solar noon, measured
    # westward. 0° at solar noon, ±15° per hour.
    hour_angle_deg = 15.0 * (solar_time_hr - 12.0)

    # Trig conversions
    lat = np.deg2rad(latitude_deg)
    dec = np.deg2rad(declination_deg)
    ha = np.deg2rad(hour_angle_deg)

    # Elevation: arcsin of the dot product of sun-direction and local-zenith vectors
    sin_elev = np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(ha)
    elevation_deg = np.rad2deg(np.arcsin(np.clip(sin_elev, -1.0, 1.0)))

    # Azimuth: more involved trig. We use the atan2 form which handles all
    # quadrants correctly (the asin form ambiguity bites everyone once).
    sin_az = -np.cos(dec) * np.sin(ha) / np.cos(np.deg2rad(elevation_deg))
    cos_az = (
        np.sin(dec) - np.sin(np.deg2rad(elevation_deg)) * np.sin(lat)
    ) / (np.cos(np.deg2rad(elevation_deg)) * np.cos(lat))
    # atan2(y, x) gives signed angle; offset by 180 to get the north=0 convention
    azimuth_deg = (np.rad2deg(np.arctan2(sin_az, cos_az)) + 180.0) % 360.0

    return pd.DataFrame(
        {
            "elevation_deg": elevation_deg,
            "azimuth_deg": azimuth_deg,
            "zenith_deg": 90.0 - elevation_deg,
        },
        index=timestamps,
    )


# ---------------------------------------------------------------------------
# Clear-sky irradiance (the "ceiling" for a given time/location)
# ---------------------------------------------------------------------------

def clear_sky_ghi(elevation_deg: np.ndarray | pd.Series) -> np.ndarray | pd.Series:
    """
    Haurwitz clear-sky model: GHI on a horizontal surface under clear skies.

    The Haurwitz model is the simplest defensible clear-sky estimate:
        G_clear = 1098 · cos(zenith) · exp(-0.057 / cos(zenith))
    It needs only the solar zenith angle — no aerosol or water vapor inputs.
    Reno et al. (2012) found it within ~10% of more elaborate models on
    cloudless days. Plenty for feature engineering; we'd use a more rigorous
    model only if we cared about clear-sky values for their own sake.

    Returns 0 when the sun is below the horizon (elevation ≤ 0).
    """
    elev = np.asarray(elevation_deg, dtype=float)
    zenith_rad = np.deg2rad(np.maximum(0.0, 90.0 - elev))
    cos_z = np.cos(zenith_rad)
    # Avoid divide-by-zero at horizon — clip cosine to a small positive value
    cos_z_safe = np.maximum(cos_z, 1e-3)
    ghi_clear = 1098.0 * cos_z_safe * np.exp(-0.057 / cos_z_safe)
    return np.where(elev > 0, ghi_clear, 0.0)


def clearness_index(
    ghi_actual: pd.Series,
    elevation_deg: pd.Series,
) -> pd.Series:
    """
    Ratio of measured GHI to clear-sky GHI.

    Bounded in roughly [0, 1.2] — values above 1 can occur briefly due to
    cloud-edge enhancement (light scattered from a cloud edge stacks with
    direct beam). The clearness index is a much more *stationary* quantity
    than raw GHI: GHI varies massively across seasons because of sun angle,
    but a "70% clear" sky has clearness ~0.7 whether it's June or December.

    For modeling, clearness index decouples "how much cloud cover is there"
    (the weather question) from "how high is the sun" (the geometry question).

    At night the ratio is genuinely undefined (0/0), but we return 0 rather
    than NaN. Returning NaN would corrupt downstream models that drop NaN rows
    — half the dataset is nighttime, and forcing models to be evaluated on
    daytime hours only (while baselines are evaluated on all hours) makes
    apples-to-oranges comparisons. The "no light" condition is already
    encoded in solar_elevation_deg, so the feature isn't lossy.
    """
    g_clear = clear_sky_ghi(elevation_deg.to_numpy())
    # At night g_clear == 0; we return 0 instead of dividing by zero.
    kt = np.where(
        g_clear > 1.0,
        ghi_actual.to_numpy() / np.where(g_clear > 1.0, g_clear, 1.0),
        0.0,
    )
    kt = np.clip(kt, 0.0, 1.5)
    return pd.Series(kt, index=ghi_actual.index, name="clearness_index")


# ---------------------------------------------------------------------------
# Plane-of-array irradiance (what the tilted panel actually receives)
# ---------------------------------------------------------------------------

def angle_of_incidence_deg(
    solar_elevation_deg: np.ndarray,
    solar_azimuth_deg: np.ndarray,
    tilt_deg: float,
    panel_azimuth_deg: float,
) -> np.ndarray:
    """
    Angle between the sun's rays and the panel's normal vector.

    cos(θ_i) = sin(elev)·cos(tilt) + cos(elev)·sin(tilt)·cos(sun_az - panel_az)

    When the sun is directly perpendicular to the panel, θ_i = 0 and the panel
    receives the maximum possible irradiance. When the sun is parallel to the
    panel surface (grazing), θ_i = 90° and the panel receives nothing direct.
    """
    elev = np.deg2rad(solar_elevation_deg)
    tilt = np.deg2rad(tilt_deg)
    az_diff = np.deg2rad(solar_azimuth_deg - panel_azimuth_deg)
    cos_aoi = np.sin(elev) * np.cos(tilt) + np.cos(elev) * np.sin(tilt) * np.cos(az_diff)
    cos_aoi = np.clip(cos_aoi, -1.0, 1.0)
    return np.rad2deg(np.arccos(cos_aoi))


def plane_of_array_irradiance(
    ghi_direct: pd.Series,
    ghi_diffuse: pd.Series,
    solar_elevation_deg: pd.Series,
    solar_azimuth_deg: pd.Series,
    tilt_deg: float,
    panel_azimuth_deg: float,
    ground_albedo: float = 0.20,
) -> pd.Series:
    """
    Compute irradiance on the tilted panel surface (W/m²).

    Three components, summed:
      - Direct (beam):  G_dir · cos(θ_i) / sin(elevation)
      - Diffuse (sky):  G_dif · (1 + cos(tilt)) / 2   [isotropic sky assumption]
      - Reflected:      (G_dir + G_dif) · albedo · (1 - cos(tilt)) / 2

    The isotropic diffuse model is the simplest one that works — it assumes
    diffuse sky radiation is uniform in all directions. More sophisticated
    models (Hay-Davies, Perez) account for circumsolar enhancement and
    horizon brightening; they're worth ~2% improvement and 10× more code.
    Worth it eventually. Not worth it for a v1 baseline.

    The 1/sin(elevation) term in the direct component arises because GHI is
    measured per horizontal surface: to recover the *beam* component, we
    divide by the cosine of the sun's zenith angle (which equals sin of
    elevation). We clip elevation to a small positive value to avoid the
    divergence at sunrise/sunset.
    """
    elev = solar_elevation_deg.to_numpy()
    sin_elev = np.sin(np.deg2rad(np.maximum(elev, 1.0)))  # clip to avoid divergence

    aoi = angle_of_incidence_deg(
        elev, solar_azimuth_deg.to_numpy(), tilt_deg, panel_azimuth_deg
    )
    cos_aoi = np.cos(np.deg2rad(aoi))
    # The beam component is only positive when sun shines on the panel front
    cos_aoi = np.maximum(cos_aoi, 0.0)

    # Beam: project the direct-normal beam onto the tilted panel
    beam = ghi_direct.to_numpy() * cos_aoi / sin_elev
    # When sun is below horizon there is no beam, regardless of what
    # the formula says
    beam = np.where(elev > 0, beam, 0.0)

    # Diffuse: isotropic sky model
    tilt_rad = np.deg2rad(tilt_deg)
    diffuse = ghi_diffuse.to_numpy() * (1 + np.cos(tilt_rad)) / 2

    # Reflected from ground (small but non-negligible for steep tilts)
    ghi_total = ghi_direct.to_numpy() + ghi_diffuse.to_numpy()
    reflected = ghi_total * ground_albedo * (1 - np.cos(tilt_rad)) / 2

    g_poa = np.maximum(beam + diffuse + reflected, 0.0)
    return pd.Series(g_poa, index=ghi_direct.index, name="g_poa")


# ---------------------------------------------------------------------------
# Cell temperature and theoretical PV output
# ---------------------------------------------------------------------------

def cell_temperature(
    ambient_temp_c: pd.Series,
    g_poa: pd.Series,
    noct_c: float = NOCT_DEG_C,
) -> pd.Series:
    """
    Estimate PV cell temperature from ambient and plane-of-array irradiance.

    NOCT (Nominal Operating Cell Temperature) model:
        T_cell = T_ambient + (NOCT - 20) · G_POA / 800

    Why this matters: hotter cells produce less power. The temperature
    coefficient is ~-0.4%/°C, so on a hot summer afternoon when T_cell hits
    50°C, the panel is already losing 10% from STC efficiency. The model
    needs this signal — without it, summer afternoons look anomalously
    underperforming.
    """
    t_cell = ambient_temp_c + (noct_c - NOCT_REFERENCE_AIR_TEMP) * g_poa / NOCT_REFERENCE_IRRADIANCE
    return t_cell.rename("t_cell_c")


def theoretical_capacity_factor(
    g_poa: pd.Series,
    t_cell_c: pd.Series,
    temp_coef: float = TEMP_COEFFICIENT_PER_C,
) -> pd.Series:
    """
    Theoretical capacity factor from physics alone — the headline feature.

    The full PV power equation is:
        P = η_STC · A · G_POA · [1 - β_T · (T_cell - 25)]

    For capacity factor (P / P_rated), the η_STC · A · G_POA_STC factors
    cancel, leaving:
        CF_theory = (G_POA / 1000) · [1 - β_T · (T_cell - 25)]

    where 1000 W/m² is the STC reference irradiance. This is a *purely
    physical* prediction of capacity factor with zero learned parameters.
    The PINN will use this as the physics prior, and learn only the
    *residual* — the gap between physics and measured grid output.

    Clipped to [0, 1] for physical realism, though the formula can produce
    slightly negative values at very hot cell temperatures (T_cell > 275 °C,
    which can't actually happen).
    """
    cf = (g_poa / 1000.0) * (1.0 - temp_coef * (t_cell_c - STC_REFERENCE_TEMP))
    return cf.clip(lower=0.0, upper=1.0).rename("cf_theory")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def add_physics_features(
    df: pd.DataFrame,
    *,
    country: str = "DE",
    site: SiteConfig | None = None,
) -> pd.DataFrame:
    """
    Add all physics-derived columns for a given country to the dataframe.

    Inputs the dataframe must contain (with `{country}_` prefix):
        {country}_temperature
        {country}_radiation_direct_horizontal
        {country}_radiation_diffuse_horizontal

    Outputs added (also prefixed by country):
        {country}_solar_elevation_deg
        {country}_solar_zenith_deg
        {country}_clearness_index
        {country}_g_poa
        {country}_t_cell_c
        {country}_cf_theory

    The country prefix lets us run this pipeline for DE training and IT
    out-of-distribution validation with the same function.
    """
    cfg = site or SiteConfig()
    df = df.copy()

    pos = solar_position(df.index, cfg.latitude_deg, cfg.longitude_deg)
    df[f"{country}_solar_elevation_deg"] = pos["elevation_deg"]
    df[f"{country}_solar_zenith_deg"] = pos["zenith_deg"]

    ghi_total = (
        df[f"{country}_radiation_direct_horizontal"]
        + df[f"{country}_radiation_diffuse_horizontal"]
    )
    df[f"{country}_clearness_index"] = clearness_index(ghi_total, pos["elevation_deg"])

    df[f"{country}_g_poa"] = plane_of_array_irradiance(
        df[f"{country}_radiation_direct_horizontal"],
        df[f"{country}_radiation_diffuse_horizontal"],
        pos["elevation_deg"],
        pos["azimuth_deg"],
        cfg.tilt_deg,
        cfg.azimuth_deg,
    )

    df[f"{country}_t_cell_c"] = cell_temperature(
        df[f"{country}_temperature"],
        df[f"{country}_g_poa"],
    )

    df[f"{country}_cf_theory"] = theoretical_capacity_factor(
        df[f"{country}_g_poa"],
        df[f"{country}_t_cell_c"],
    )

    return df
