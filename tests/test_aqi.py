"""Tests for the EPA AQI computation.

This is the most domain-specific logic in the project -- breakpoint tables,
truncation rules and unit conversions -- and it feeds every downstream label,
so a silent error here corrupts the whole dataset rather than crashing.
"""

import pytest

from feature_pipeline.aqi import _sub_index, _ugm3_to_ppb, aqi_category, compute_aqi

# A benign real reading, used as a base that individual tests perturb.
CLEAN_COMPONENTS = {
    "co": 342.12, "no": 0.14, "no2": 2.1, "o3": 60.0,
    "so2": 1.69, "pm2_5": 5.0, "pm10": 10.0, "nh3": 5.92,
}


class TestSubIndexBreakpoints:
    @pytest.mark.parametrize(
        "pollutant, concentration, expected",
        [
            ("pm2_5", 0.0, 0),      # bottom of scale
            ("pm2_5", 9.0, 50),     # top of "Good" (2024-revised breakpoint)
            ("pm2_5", 9.1, 51),     # bottom of "Moderate"
            ("pm2_5", 35.4, 100),
            ("pm10", 54, 50),
            ("pm10", 55, 51),
        ],
    )
    def test_breakpoint_edges_map_to_exact_aqi(self, pollutant, concentration, expected):
        assert _sub_index(pollutant, concentration) == expected

    @pytest.mark.parametrize("pollutant", ["pm2_5", "pm10", "o3", "co", "so2", "no2"])
    def test_negative_concentration_scores_zero_not_none(self, pollutant):
        """OpenWeather's transport model emits small negative concentrations for
        trace gases. EPA breakpoints start at 0, so before clamping these fell
        through every range and returned None, which crashed compute_aqi's
        max(). Found when the backfill widened from 90 days to 5.7 years."""
        assert _sub_index(pollutant, -0.5) == 0

    @pytest.mark.parametrize("pollutant", ["pm2_5", "pm10", "so2", "no2"])
    def test_concentration_above_top_breakpoint_pins_to_500(self, pollutant):
        assert _sub_index(pollutant, 10_000) == 500

    def test_value_in_gap_between_breakpoints_is_truncated_not_dropped(self):
        """PM10 buckets end at 54 and resume at 55; 54.7 must truncate into the
        lower bucket rather than matching nothing."""
        assert _sub_index("pm10", 54.7) == 50


class TestUnitConversion:
    def test_ugm3_to_ppb_matches_standard_formula(self):
        # ppb = ug/m3 * 24.45 / MW at 25C, 1 atm
        assert _ugm3_to_ppb(100.0, 48.00) == pytest.approx(50.9375, rel=1e-4)


class TestComputeAqi:
    def test_overall_aqi_is_the_worst_sub_index(self):
        components = dict(CLEAN_COMPONENTS, pm2_5=150.0)  # clearly the worst
        aqi, dominant, sub_indices = compute_aqi(components)
        assert dominant == "pm2_5"
        assert aqi == max(sub_indices.values())

    def test_negative_readings_do_not_crash(self):
        components = dict(CLEAN_COMPONENTS, no2=-0.01, so2=-1.2)
        aqi, _, _ = compute_aqi(components)
        assert 0 <= aqi <= 500

    def test_non_aqi_pollutants_are_ignored(self):
        """nh3 and no are stored as features but are not EPA AQI pollutants."""
        _, _, sub_indices = compute_aqi(CLEAN_COMPONENTS)
        assert "nh3" not in sub_indices and "no" not in sub_indices

    def test_unscoreable_input_raises_rather_than_inventing_a_value(self):
        with pytest.raises(ValueError):
            compute_aqi({"nh3": 1.0, "no": 2.0})


class TestCategories:
    @pytest.mark.parametrize(
        "aqi, name, severity",
        [
            (0, "Good", None),
            (50, "Good", None),
            (51, "Moderate", None),
            (101, "Unhealthy for Sensitive Groups", "warning"),
            (151, "Unhealthy", "serious"),
            (201, "Very Unhealthy", "critical"),
            (301, "Hazardous", "critical"),
        ],
    )
    def test_category_boundaries(self, aqi, name, severity):
        assert aqi_category(aqi)[:2] == (name, severity)

    def test_above_scale_stays_hazardous(self):
        assert aqi_category(750)[0] == "Hazardous"
