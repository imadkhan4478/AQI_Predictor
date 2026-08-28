"""Tests for the human-readable feature labels.

The explainability panel printed raw column names -- `aqi_rmean_72`, `hour_sin` --
which tell a reader nothing and read as unfinished work.
"""

from training_pipeline.data import FEATURE_COLUMNS
from training_pipeline.feature_names import display_name


def test_lags_and_windows_read_as_english():
    assert display_name("aqi_lag_24") == "AQI 24h ago"
    assert display_name("aqi_rmean_72") == "AQI 72h average"
    assert display_name("aqi_rstd_3") == "AQI 3h variability"


def test_pollutants_carry_their_formula():
    assert display_name("pm2_5") == "PM2.5"
    assert display_name("o3") == "Ozone (O3)"


def test_every_trained_feature_has_a_label():
    """A raw column name reaching the chart is the defect this module exists to fix."""
    for column in FEATURE_COLUMNS:
        label = display_name(column)
        assert label
        assert "_" not in label, f"{column} still renders as a column name: {label}"


def test_an_unknown_column_degrades_readably_rather_than_raising():
    """A new feature should appear under a tolerable name, not break the panel."""
    assert display_name("some_new_thing") == "Some new thing"
