"""Tests for the dashboard's presentation helpers.

Streamlit is imported but never run: only the pure functions are exercised, so
these need no server and no credentials.
"""

from datetime import datetime, timedelta, timezone

import pytest

from app import app as dash
from feature_pipeline.aqi import _CATEGORIES


def payload_observed_hours_ago(hours):
    observed = datetime.now(timezone.utc) - timedelta(hours=hours)
    return {"observed_at": observed.replace(microsecond=0).isoformat()}


class TestContrast:
    def test_yellow_gets_dark_text(self):
        assert dash.readable_text_color("#ffff00") == "#111111"

    def test_dark_red_gets_light_text(self):
        assert dash.readable_text_color("#7e0023") == "#ffffff"

    def test_every_epa_category_is_legible(self):
        """Not a snapshot of today's palette -- an assertion that whatever colour a
        category carries, the text on it is the higher-contrast of black or white."""
        for *_, color in _CATEGORIES:
            chosen = dash.readable_text_color(color)
            assert chosen in ("#111111", "#ffffff")
            assert contrast_ratio(color, chosen) >= contrast_ratio(
                color, "#ffffff" if chosen == "#111111" else "#111111"
            )


def contrast_ratio(background_hex, text_hex):
    lighter, darker = sorted((_luminance(background_hex), _luminance(text_hex)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def _luminance(hex_color):
    channels = [int(hex_color.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


class TestObservationAge:
    def test_age_is_measured_from_now(self):
        age_hours, _, _ = dash.observation_age(payload_observed_hours_ago(5))
        assert 4.9 < age_hours < 5.1

    def test_timestamp_is_rendered_in_local_time(self):
        _, local_text, tz_label = dash.observation_age(
            {"observed_at": "2026-08-26T02:00:00+00:00"}
        )
        # Asia/Karachi is UTC+5, so 02:00 UTC is 07:00 the same day.
        assert "07:00" in local_text
        assert "26 August 2026" in local_text
        assert tz_label

    @pytest.mark.parametrize(
        "hours, expected",
        [(0.2, "less than an hour ago"), (1.0, "less than an hour ago"),
         (3.0, "3 hours ago"), (30.0, "30 hours ago"), (264.0, "11 days ago")],
    )
    def test_relative_phrasing(self, hours, expected):
        assert dash.relative_age(hours) == expected

    def test_the_eleven_day_incident_reads_as_days(self):
        """2026-08-15 02:00 UTC served on 2026-08-26 -- the reading that went
        unnoticed because the page printed a raw ISO timestamp."""
        assert dash.relative_age(265.0) == "11 days ago"


class TestDegradation:
    """The page must never answer a remote failure with a traceback."""

    def test_successful_payload_is_retained(self, monkeypatch):
        dash.last_good_payload().clear()
        monkeypatch.setattr(dash, "fetch_payload", lambda city: {"observed_at": "x"})
        payload, live = dash.fetch_payload_or_last_good("Lahore")
        assert live is True
        assert dash.last_good_payload()["payload"] == payload

    def test_retained_payload_is_served_when_the_read_fails(self, monkeypatch):
        dash.last_good_payload().clear()
        monkeypatch.setattr(dash, "fetch_payload", lambda city: {"observed_at": "good"})
        dash.fetch_payload_or_last_good("Lahore")

        def boom(city):
            raise RuntimeError("Query Service refused the read")

        monkeypatch.setattr(dash, "fetch_payload", boom)
        payload, live = dash.fetch_payload_or_last_good("Lahore")
        assert live is False
        assert payload["observed_at"] == "good"

    def test_failure_with_nothing_retained_propagates(self, monkeypatch):
        dash.last_good_payload().clear()

        def boom(city):
            raise RuntimeError("Query Service refused the read")

        monkeypatch.setattr(dash, "fetch_payload", boom)
        with pytest.raises(RuntimeError):
            dash.fetch_payload_or_last_good("Lahore")


class TestEvaluationPanel:
    """The walk-forward comparison is the project's strongest claim; it has to
    reach the page correctly and it has to be honest about partial wins."""

    def details(self, metrics, horizon=24, version=7, weight=0.35):
        return [
            {
                "horizon_hours": horizon,
                "model_name": f"aqi_forecast_{horizon}h",
                "version": version,
                "blend_weight": weight,
                "metrics": metrics,
            }
        ]

    def test_a_clear_win_lists_every_metric(self):
        wins = dash.beats_baseline(
            {
                "R2": 0.715, "persistence_R2": 0.628,
                "MAE": 24.0, "persistence_MAE": 30.0,
                "RMSE": 40.0, "persistence_RMSE": 52.0,
            }
        )
        assert wins == ["R2", "MAE", "RMSE"]

    def test_the_real_24h_result_does_not_claim_an_mae_win(self):
        """Production 24h: R2 0.832 vs 0.814, but MAE 30.2 against 29.8 -- worse.
        The panel must not round that into "beats the baseline"."""
        wins = dash.beats_baseline(
            {
                "R2": 0.832, "persistence_R2": 0.814,
                "MAE": 30.2, "persistence_MAE": 29.8,
                "RMSE": 45.0, "persistence_RMSE": 48.0,
            }
        )
        assert wins == ["R2", "RMSE"]
        assert "MAE" not in wins

    def test_a_missing_baseline_is_never_counted_as_a_win(self):
        """persistence_RMSE does not exist on models registered before it was
        added; absent evidence is not evidence."""
        wins = dash.beats_baseline({"R2": 0.8, "persistence_R2": 0.7, "RMSE": 40.0})
        assert wins == ["R2"]

    def test_losing_on_everything_says_so(self):
        wins = dash.beats_baseline(
            {
                "R2": 0.5, "persistence_R2": 0.7,
                "MAE": 40.0, "persistence_MAE": 30.0,
                "RMSE": 60.0, "persistence_RMSE": 50.0,
            }
        )
        assert wins == []
        row = dash.evaluation_rows(self.details({
            "R2": 0.5, "persistence_R2": 0.7,
            "MAE": 40.0, "persistence_MAE": 30.0,
            "RMSE": 60.0, "persistence_RMSE": 50.0,
        }))[0]
        assert row["Beats baseline on"] == "nothing"

    def test_row_carries_the_baseline_beside_the_model(self):
        row = dash.evaluation_rows(self.details({
            "R2": 0.832, "persistence_R2": 0.814,
            "MAE": 30.2, "persistence_MAE": 29.8,
            "RMSE": 45.0, "persistence_RMSE": 48.0,
        }))[0]
        assert row["Horizon"] == "24h"
        assert row["R2"] == "0.832" and row["R2 baseline"] == "0.814"
        assert row["MAE"] == "30.2" and row["MAE baseline"] == "29.8"
        assert row["Blend weight"] == "0.35"
        assert row["Version"] == 7

    def test_missing_numbers_render_as_a_dash_not_a_zero(self):
        row = dash.evaluation_rows(self.details({}))[0]
        assert row["R2"] == "—" and row["RMSE baseline"] == "—"

    def test_an_unregistered_weight_renders_as_a_dash(self):
        row = dash.evaluation_rows(self.details({}, weight=None))[0]
        assert row["Blend weight"] == "—"


class TestRegistryKeyMangling:
    """Hopsworks rewrites metric keys on the way in. These are the keys actually
    read back from the registry on 2026-08-27 -- note `persistence__r2` with two
    underscores while top-level `MAE` and `R2` survive untouched. An exact-match
    lookup blanked every baseline column and reported that a model beating its
    baseline beat nothing."""

    AS_STORED = {
        "MAE": 30.238176677122784,
        "R2": 0.8316838404747994,
        "blend_weight": 0.35000000000000003,
        "persistence_mae": 29.838013799343965,
        "persistence__r2": 0.8140342233136757,
        "RMSE": 48.35603293806847,
    }

    def test_the_mangled_baseline_keys_are_found(self):
        assert dash.metric(self.AS_STORED, "persistence_R2") == 0.8140342233136757
        assert dash.metric(self.AS_STORED, "persistence_MAE") == 29.838013799343965

    def test_unmangled_keys_still_resolve(self):
        assert dash.metric(self.AS_STORED, "R2") == 0.8316838404747994
        assert dash.metric(self.AS_STORED, "MAE") == 30.238176677122784

    def test_a_genuinely_absent_metric_is_none(self):
        """persistence_RMSE was added later, so these versions do not carry it."""
        assert dash.metric(self.AS_STORED, "persistence_RMSE") is None

    def test_the_real_row_renders_the_real_comparison(self):
        row = dash.evaluation_rows(
            [
                {
                    "horizon_hours": 24,
                    "model_name": "aqi_forecast_24h",
                    "version": 7,
                    "blend_weight": 0.35,
                    "metrics": self.AS_STORED,
                }
            ]
        )[0]
        assert row["R2"] == "0.832" and row["R2 baseline"] == "0.814"
        assert row["MAE"] == "30.2" and row["MAE baseline"] == "29.8"
        assert row["RMSE baseline"] == "—"
        # Wins R2, loses MAE, RMSE not comparable -- and says exactly that.
        assert row["Beats baseline on"] == "R2"

    def test_no_baseline_at_all_is_distinguished_from_losing(self):
        """"The blend lost" and "nobody wrote the comparison down" must not look
        alike, for the same reason a missing blend weight withholds a horizon."""
        absent = dash.evaluation_rows(
            [{"horizon_hours": 24, "model_name": "m", "version": 1,
              "blend_weight": 0.35, "metrics": {"R2": 0.8, "MAE": 30.0}}]
        )[0]
        assert absent["Beats baseline on"] == "no baseline recorded"

        lost = dash.evaluation_rows(
            [{"horizon_hours": 24, "model_name": "m", "version": 1, "blend_weight": 0.35,
              "metrics": {"R2": 0.5, "persistence__r2": 0.7}}]
        )[0]
        assert lost["Beats baseline on"] == "nothing"


class TestForecastDates:
    """"+24h" is a horizon, not a time. A reader planning tomorrow needs the date."""

    def payload(self):
        return {
            "observed_at": "2026-08-27T08:00:00+00:00",
            "current": {"aqi": 148, "category": "Unhealthy for Sensitive Groups",
                        "color": "#ff7e00", "severity": "warning"},
            "forecast": [
                {"horizon_hours": 24, "aqi": 143, "category": "Unhealthy for Sensitive Groups",
                 "color": "#ff7e00", "severity": "warning"},
                {"horizon_hours": 72, "aqi": 141, "category": "Unhealthy for Sensitive Groups",
                 "color": "#ff7e00", "severity": "warning"},
            ],
        }

    def test_each_horizon_becomes_a_wall_clock_time(self):
        points = dash.forecast_points(self.payload())
        assert points[0]["at"].isoformat() == "2026-08-28T08:00:00+00:00"
        assert points[1]["at"].isoformat() == "2026-08-30T08:00:00+00:00"

    def test_the_anchor_is_the_observation_not_now(self):
        """Anchoring to the current clock would date a forecast made from an
        11-hour-old reading as though it started now."""
        points = dash.forecast_points(self.payload())
        assert points[0]["at"] - datetime.fromisoformat("2026-08-27T08:00:00+00:00") == timedelta(
            hours=24
        )


class TestExplainabilityPanel:
    RANKING = {
        "horizon_hours": 24,
        "explained_rows": 2000,
        "features": [
            {"column": "aqi", "label": "Current AQI", "mean_abs_shap": 12.5},
            {"column": "aqi_rmean_72", "label": "AQI 72h average", "mean_abs_shap": 4.1},
            {"column": "o3", "label": "Ozone (O3)", "mean_abs_shap": 2.2},
        ],
    }

    def test_the_chart_uses_labels_not_column_names(self):
        figure = dash.plot_shap(self.RANKING)
        labels = list(figure.data[0].y)
        assert "Current AQI" in labels
        assert "aqi_rmean_72" not in labels

    def test_the_largest_contributor_sits_at_the_top(self):
        """Plotly draws a horizontal bar chart bottom-up, so the ranking is reversed
        on the way in and the biggest bar must end up last."""
        figure = dash.plot_shap(self.RANKING)
        assert figure.data[0].y[-1] == "Current AQI"

    def test_the_column_name_is_still_available_on_hover(self):
        """A reader gets the readable label; a developer still needs the real column."""
        figure = dash.plot_shap(self.RANKING)
        assert "aqi_rmean_72" in list(figure.data[0].customdata)

    def test_a_missing_ranking_file_is_not_an_error(self, monkeypatch, tmp_path):
        monkeypatch.setattr(dash, "SHAP_RANKING_PATH", str(tmp_path / "absent.json"))
        dash.shap_ranking.clear()
        assert dash.shap_ranking() is None


class TestReadingCards:
    def test_the_card_carries_the_category_colour_as_its_accent(self):
        html = dash.reading_card(148, "Unhealthy for Sensitive Groups", "#ff7e00", "Observed")
        assert "--accent:#ff7e00" in html
        assert "148" in html and "Unhealthy for Sensitive Groups" in html

    def test_the_current_reading_is_emphasised(self):
        assert 'class="reading now"' in dash.reading_card(1, "Good", "#0f0", "x", emphasis=True)
        assert 'class="reading "' in dash.reading_card(1, "Good", "#0f0", "x")

    def test_error_is_shown_to_one_decimal_so_a_narrow_loss_stays_visible(self):
        """At whole-AQI precision the real 24h pair reads "±30 · baseline ±30",
        which hides that the blend is worse than persistence on MAE."""
        text = dash.typical_error_text({"MAE": 30.238, "persistence_mae": 29.838})
        assert "±30.2" in text and "±29.8" in text

    def test_a_missing_baseline_is_not_invented(self):
        assert dash.typical_error_text({"MAE": 30.2}) == "Typical error ±30.2 AQI"

    def test_a_version_without_metrics_says_so(self):
        assert "not recorded" in dash.typical_error_text({})


class TestEmptyBaselineColumns:
    """persistence_RMSE is only stored on versions registered after it was added,
    so that column is uniformly blank until the next retrain. An empty column reads
    as unfinished work."""

    def rows(self, rmse_baseline="—"):
        return [
            {"Horizon": "24h", "R2": "0.832", "R2 baseline": "0.814",
             "RMSE": "48.4", "RMSE baseline": rmse_baseline},
            {"Horizon": "48h", "R2": "0.748", "R2 baseline": "0.709",
             "RMSE": "59.0", "RMSE baseline": "—"},
        ]

    def test_a_uniformly_empty_baseline_column_is_dropped_and_named(self):
        trimmed, missing = dash.drop_empty_baselines(self.rows())
        assert "RMSE baseline" not in trimmed[0]
        assert missing == ["RMSE"]
        # The model's own RMSE stays: only the absent comparison goes.
        assert trimmed[0]["RMSE"] == "48.4"

    def test_a_partially_populated_column_is_kept(self):
        """One version carrying the figure is reason to show the column, not hide it."""
        trimmed, missing = dash.drop_empty_baselines(self.rows(rmse_baseline="52.1"))
        assert "RMSE baseline" in trimmed[0]
        assert missing == []

    def test_populated_columns_are_never_dropped(self):
        trimmed, missing = dash.drop_empty_baselines(self.rows())
        assert trimmed[0]["R2 baseline"] == "0.814"
        assert "R2" not in missing


class TestFreshnessThresholds:
    """How loudly is a separate question from whether. The age is always stated;
    the escalation is calibrated to this pipeline's real cadence."""

    def test_a_few_hours_is_normal_operation_not_a_fault(self):
        """GitHub delivers the hourly schedule 3-7 times a day, so warning at two
        hours put a warning on a healthy system."""
        assert dash.freshness_level(0.5) == "current"
        assert dash.freshness_level(3.0) == "current"
        assert dash.freshness_level(5.9) == "current"

    def test_past_the_pipeline_tolerance_it_is_late(self):
        assert dash.freshness_level(6.0) == "late"
        assert dash.freshness_level(23.9) == "late"

    def test_past_a_day_the_page_is_describing_yesterday(self):
        assert dash.freshness_level(24.0) == "broken"
        assert dash.freshness_level(265.0) == "broken"

    def test_the_dashboard_threshold_matches_the_pipeline_tolerance(self):
        """Two components judging the same store by different rules is how the
        dashboard and the pipeline disagreed about staleness in the first place."""
        from feature_pipeline.pipeline import STALE_AFTER_HOURS as pipeline_tolerance

        assert dash.STALE_AFTER_HOURS == pipeline_tolerance


def test_the_dominant_pollutant_is_named_not_column_cased():
    """`pm2_5` in the sidebar is the same defect as `aqi_rmean_72` on a chart axis."""
    from training_pipeline.feature_names import display_name

    assert display_name("pm2_5") == "PM2.5"
    assert display_name("o3") == "Ozone (O3)"
