"""Shared Hopsworks connection logic, used by both the live pipeline and backfill."""

import os
import tempfile

import hopsworks
from hopsworks_common.client.exceptions import RestAPIError

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 2


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
        description=(
            "Hourly weather + pollutant readings with computed AQI, "
            "time-based features (hour/day/month), and AQI change rate"
        ),
        primary_key=["city_name", "timestamp"],
        event_time="timestamp",
        time_travel_format="HUDI",
    )


def read_features_df():
    """Read all rows currently stored in the feature group as a pandas DataFrame."""
    fg = get_feature_group()
    return fg.read()


def get_aqi_at(city_name, timestamp):
    """AQI recorded for this city at exactly `timestamp`, or None if absent.

    Callers use this to build aqi_change_rate against the immediately
    preceding hour. Returning None for a missing hour (rather than falling
    back to whatever row happened to be latest) keeps the feature meaning the
    same in production as it does in the backfill, where rows are contiguous.

    Note the narrow except: a genuinely empty feature group is expected on a
    fresh feature-group version, but an auth or permission failure must not be
    silently converted into "no previous reading" -- that would write a wrong
    change rate into the store with no error surfaced anywhere.
    """
    try:
        df = read_features_df()
    except RestAPIError as error:
        if _is_empty_feature_group(error):
            return None
        raise

    match = df[(df["city_name"] == city_name) & (df["timestamp"] == timestamp)]
    if match.empty:
        return None
    return match.iloc[-1]["aqi"]


def _is_empty_feature_group(error):
    """Hopsworks raises a generic RestAPIError when a feature group exists but
    has no materialised data yet, which is a normal state right after creating
    a new feature group version."""
    message = str(error).lower()
    return "no data" in message or "not found" in message or "empty" in message
