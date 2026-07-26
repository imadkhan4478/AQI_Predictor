"""Shared Hopsworks connection logic, used by both the live pipeline and backfill."""

import os
import tempfile

import hopsworks

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1


def get_feature_group():
    project = hopsworks.login(
        project=os.environ["HOPSWORKS_PROJECT_NAME"],
        api_key_value=os.environ["HOPSWORKS_API_KEY"],
        cert_folder=os.path.join(tempfile.gettempdir(), "hopsworks_certs"),
    )
    fs = project.get_feature_store()
    return fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description="Hourly weather + pollutant readings with computed AQI",
        primary_key=["city_name", "timestamp"],
        event_time="timestamp",
        time_travel_format="HUDI",
    )


def read_features_df():
    """Read all rows currently stored in the feature group as a pandas DataFrame."""
    fg = get_feature_group()
    return fg.read()
