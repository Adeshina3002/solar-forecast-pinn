"""
Unit tests for the physics module.

Physics code is the best kind to unit-test because *correct answers exist*
independent of the code. We can verify against textbook reference cases:
- The sun at solar noon on the equator on the equinox is directly overhead
- Solar elevation at the North Pole on the December solstice is below zero
- The angle of incidence is zero when sun rays are perpendicular to the panel

A failure in any of these is unambiguous — it means the formula is wrong,
not that some heuristic is off.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from solar_forecast.physics import (
    SiteConfig,
    add_physics_features,
    angle_of_incidence_deg,
    cell_temperature,
    clear_sky_ghi,
    clearness_index,
    plane_of_array_irradiance,
    solar_position,
    theoretical_capacity_factor,
)


# ---------------------------------------------------------------------------
# Solar position
# ---------------------------------------------------------------------------

class TestSolarPosition:
    def test_sun_overhead_at_equator_equinox_noon(self):
        """March 20 (~vernal equinox), 12:00 UTC, at 0°N 0°E:
        sun should be ~directly overhead (elevation ~90°, zenith ~0°)."""
        idx = pd.DatetimeIndex(["2018-03-20 12:00:00"], tz="UTC")
        pos = solar_position(idx, latitude_deg=0.0, longitude_deg=0.0)
        # Allow ~3° tolerance — declination on exact equinox isn't exactly 0
        # due to the simple Spencer formula, and equation-of-time corrections
        # shift solar noon by up to ~15 min.
        assert pos["elevation_deg"].iloc[0] > 87.0

    def test_sun_below_horizon_at_north_pole_winter(self):
        """North Pole, December solstice, polar night → elevation negative."""
        idx = pd.DatetimeIndex(["2018-12-21 12:00:00"], tz="UTC")
        pos = solar_position(idx, latitude_deg=90.0, longitude_deg=0.0)
        assert pos["elevation_deg"].iloc[0] < 0.0

    def test_germany_midday_summer_high(self):
        """Berlin (~52.5°N), 12:00 UTC on summer solstice: elevation should
        be high — ~61° (90 - 52.5 + 23.5)."""
        idx = pd.DatetimeIndex(["2018-06-21 12:00:00"], tz="UTC")
        pos = solar_position(idx, latitude_deg=52.5, longitude_deg=13.4)
        assert 55.0 < pos["elevation_deg"].iloc[0] < 65.0

    def test_germany_midnight_summer_below_horizon(self):
        """At Berlin, 00:00 UTC even in summer: sun must be below horizon."""
        idx = pd.DatetimeIndex(["2018-06-21 00:00:00"], tz="UTC")
        pos = solar_position(idx, latitude_deg=52.5, longitude_deg=13.4)
        assert pos["elevation_deg"].iloc[0] < 0.0

    def test_requires_timezone_aware_index(self):
        """tz-naive timestamps should raise — silent UTC conversion is a bug magnet."""
        idx = pd.DatetimeIndex(["2018-06-21 12:00:00"])  # no tz
        with pytest.raises(ValueError, match="tz-aware"):
            solar_position(idx, latitude_deg=51.0, longitude_deg=10.5)


# ---------------------------------------------------------------------------
# Clear-sky GHI
# ---------------------------------------------------------------------------

class TestClearSky:
    def test_zero_when_sun_below_horizon(self):
        """No sun above horizon = no irradiance, full stop."""
        ghi = clear_sky_ghi(np.array([-5.0, -1.0, 0.0]))
        assert np.all(ghi == 0.0)

    def test_maximum_near_solar_zenith(self):
        """Clear-sky GHI peaks when sun is directly overhead.
        At 90° elevation, the Haurwitz model gives ~1098 · 1 · exp(-0.057) ≈ 1037 W/m²."""
        ghi = clear_sky_ghi(np.array([90.0]))[0]
        assert 1000.0 < ghi < 1100.0

    def test_monotonic_increase_with_elevation(self):
        """Higher sun → more clear-sky irradiance, always."""
        elevs = np.array([10.0, 30.0, 60.0, 90.0])
        ghis = clear_sky_ghi(elevs)
        assert np.all(np.diff(ghis) > 0)


# ---------------------------------------------------------------------------
# Clearness index
# ---------------------------------------------------------------------------

class TestClearnessIndex:
    def test_clear_sky_gives_near_one(self):
        """If measured GHI equals clear-sky GHI, clearness index = 1."""
        elev = pd.Series([45.0], index=pd.DatetimeIndex(["2018-06-21 12:00", ]))
        ghi_clear = clear_sky_ghi(elev.to_numpy())
        ghi_actual = pd.Series(ghi_clear, index=elev.index)
        kt = clearness_index(ghi_actual, elev)
        assert kt.iloc[0] == pytest.approx(1.0, abs=0.01)

    def test_overcast_gives_low_clearness(self):
        """50% of clear-sky → clearness ~0.5."""
        elev = pd.Series([45.0], index=pd.DatetimeIndex(["2018-06-21 12:00"]))
        ghi_clear = clear_sky_ghi(elev.to_numpy())
        ghi_actual = pd.Series(ghi_clear * 0.5, index=elev.index)
        kt = clearness_index(ghi_actual, elev)
        assert kt.iloc[0] == pytest.approx(0.5, abs=0.01)

    def test_night_returns_zero(self):
        """At night the ratio is undefined, but we return 0 rather than NaN
        — see docstring of clearness_index for the rationale."""
        elev = pd.Series([-10.0], index=pd.DatetimeIndex(["2018-06-21 00:00"]))
        ghi_actual = pd.Series([0.0], index=elev.index)
        kt = clearness_index(ghi_actual, elev)
        assert kt.iloc[0] == 0.0


# ---------------------------------------------------------------------------
# Angle of incidence
# ---------------------------------------------------------------------------

class TestAngleOfIncidence:
    def test_zero_when_sun_normal_to_panel(self):
        """Sun directly perpendicular to panel → AOI = 0°.
        Sun at elev=55° in the south (180°), panel tilted 35° facing south:
        the sun is exactly normal to the panel."""
        aoi = angle_of_incidence_deg(
            solar_elevation_deg=np.array([55.0]),
            solar_azimuth_deg=np.array([180.0]),
            tilt_deg=35.0,
            panel_azimuth_deg=180.0,
        )
        assert aoi[0] == pytest.approx(0.0, abs=0.5)

    def test_ninety_when_sun_at_horizon_south_panel_north(self):
        """Sun on horizon to the south, panel facing north — AOI should be 90°
        (sun rays parallel to panel surface, no direct gain)."""
        aoi = angle_of_incidence_deg(
            solar_elevation_deg=np.array([0.0]),
            solar_azimuth_deg=np.array([180.0]),
            tilt_deg=90.0,
            panel_azimuth_deg=0.0,
        )
        # Should be close to 180° (sun behind the panel) — direct gain is zero
        # regardless. We just check it's not ~0°.
        assert aoi[0] > 90.0


# ---------------------------------------------------------------------------
# Plane-of-array irradiance
# ---------------------------------------------------------------------------

class TestPlaneOfArrayIrradiance:
    def test_zero_when_sun_below_horizon(self):
        """Night → G_POA = 0 regardless of GHI inputs."""
        idx = pd.DatetimeIndex(["2018-12-21 00:00"], tz="UTC")
        g_poa = plane_of_array_irradiance(
            ghi_direct=pd.Series([0.0], index=idx),
            ghi_diffuse=pd.Series([0.0], index=idx),
            solar_elevation_deg=pd.Series([-10.0], index=idx),
            solar_azimuth_deg=pd.Series([0.0], index=idx),
            tilt_deg=35.0,
            panel_azimuth_deg=180.0,
        )
        assert g_poa.iloc[0] == 0.0

    def test_horizontal_panel_recovers_ghi(self):
        """A panel with 0° tilt should receive ~GHI (it IS horizontal).
        Some divergence near horizon expected — test at high sun."""
        idx = pd.DatetimeIndex(["2018-06-21 12:00"], tz="UTC")
        ghi_dir = pd.Series([700.0], index=idx)
        ghi_dif = pd.Series([100.0], index=idx)
        g_poa = plane_of_array_irradiance(
            ghi_direct=ghi_dir,
            ghi_diffuse=ghi_dif,
            solar_elevation_deg=pd.Series([60.0], index=idx),
            solar_azimuth_deg=pd.Series([180.0], index=idx),
            tilt_deg=0.0,
            panel_azimuth_deg=180.0,
        )
        # At tilt=0, beam projection = ghi_dir/sin(elev) · cos(zenith) = ghi_dir
        # diffuse fully visible (1+cos(0))/2 = 1, reflected = 0
        # So G_POA should equal GHI total
        assert g_poa.iloc[0] == pytest.approx(800.0, rel=0.02)


# ---------------------------------------------------------------------------
# Cell temperature & theoretical CF
# ---------------------------------------------------------------------------

class TestCellTemperature:
    def test_equals_ambient_when_no_irradiance(self):
        """No sun → cell same temperature as ambient air."""
        idx = pd.DatetimeIndex(["2018-12-21 00:00"], tz="UTC")
        t_amb = pd.Series([10.0], index=idx)
        g_poa = pd.Series([0.0], index=idx)
        t_cell = cell_temperature(t_amb, g_poa)
        assert t_cell.iloc[0] == 10.0

    def test_hotter_under_irradiance(self):
        """Sun on the panel → cell hotter than ambient."""
        idx = pd.DatetimeIndex(["2018-06-21 12:00"], tz="UTC")
        t_amb = pd.Series([20.0], index=idx)
        g_poa = pd.Series([800.0], index=idx)  # NOCT reference irradiance
        t_cell = cell_temperature(t_amb, g_poa)
        # At G=800 W/m² the formula gives T_cell = T_amb + (45 - 20) = T_amb + 25
        assert t_cell.iloc[0] == pytest.approx(45.0, abs=0.1)


class TestTheoreticalCapacityFactor:
    def test_zero_in_darkness(self):
        idx = pd.DatetimeIndex(["2018-12-21 00:00"], tz="UTC")
        cf = theoretical_capacity_factor(
            g_poa=pd.Series([0.0], index=idx),
            t_cell_c=pd.Series([10.0], index=idx),
        )
        assert cf.iloc[0] == 0.0

    def test_near_one_at_stc(self):
        """At Standard Test Conditions (G_POA=1000, T_cell=25), CF should equal 1."""
        idx = pd.DatetimeIndex(["2018-06-21 12:00"], tz="UTC")
        cf = theoretical_capacity_factor(
            g_poa=pd.Series([1000.0], index=idx),
            t_cell_c=pd.Series([25.0], index=idx),
        )
        assert cf.iloc[0] == pytest.approx(1.0, abs=0.01)

    def test_temperature_derating(self):
        """At G=1000 but T_cell=50°C, CF should be reduced by ~10%.
        β·ΔT = 0.004 · 25 = 0.10 → CF = 0.90."""
        idx = pd.DatetimeIndex(["2018-06-21 12:00"], tz="UTC")
        cf = theoretical_capacity_factor(
            g_poa=pd.Series([1000.0], index=idx),
            t_cell_c=pd.Series([50.0], index=idx),
        )
        assert cf.iloc[0] == pytest.approx(0.90, abs=0.01)


# ---------------------------------------------------------------------------
# add_physics_features (integration)
# ---------------------------------------------------------------------------

class TestAddPhysicsFeatures:
    def test_adds_all_expected_columns(self):
        """The orchestration function should add all six derived columns."""
        idx = pd.date_range("2018-06-21", periods=24, freq="h", tz="UTC")
        df = pd.DataFrame({
            "DE_temperature": np.full(24, 20.0),
            "DE_radiation_direct_horizontal": np.full(24, 500.0),
            "DE_radiation_diffuse_horizontal": np.full(24, 100.0),
        }, index=idx)
        out = add_physics_features(df, country="DE")
        for col in [
            "DE_solar_elevation_deg",
            "DE_solar_zenith_deg",
            "DE_clearness_index",
            "DE_g_poa",
            "DE_t_cell_c",
            "DE_cf_theory",
        ]:
            assert col in out.columns, f"missing {col}"

    def test_cf_theory_never_exceeds_one(self):
        """CF_theory must be physically valid (in [0, 1]) for all hours."""
        idx = pd.date_range("2018-06-21", periods=24, freq="h", tz="UTC")
        df = pd.DataFrame({
            "DE_temperature": np.linspace(15, 30, 24),
            "DE_radiation_direct_horizontal": np.linspace(0, 800, 24),
            "DE_radiation_diffuse_horizontal": np.linspace(0, 200, 24),
        }, index=idx)
        out = add_physics_features(df, country="DE")
        assert out["DE_cf_theory"].min() >= 0.0
        assert out["DE_cf_theory"].max() <= 1.0
