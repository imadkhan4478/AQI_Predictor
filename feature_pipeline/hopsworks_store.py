"""Shared Hopsworks connection logic, used by both the live pipeline and backfill."""

import os
import tempfile
import time

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


MAX_READ_ATTEMPTS = 3


def _read_with_retry(query, description):
    """Feature-store reads go over Arrow Flight (gRPC), which drops connections
    intermittently -- especially on larger transfers. These are transient, so
    retry before giving up and failing the whole pipeline run."""
    last_error = None
    for attempt in range(1, MAX_READ_ATTEMPTS + 1):
        try:
            return query()
        except Exception as error:  # FlightUnavailableError et al. aren't a shared base class
            last_error = error
            if attempt == MAX_READ_ATTEMPTS:
                raise
            wait_seconds = attempt * 5
            print(
                f"{description} failed ({type(error).__name__}), "
                f"attempt {attempt}/{MAX_READ_ATTEMPTS}. Retrying in {wait_seconds}s...",
                flush=True,
            )
            time.sleep(wait_seconds)
    raise last_error


def read_features_df():
    """Read all rows currently stored in the feature group as a pandas DataFrame.

    Only for training and analysis, which genuinely need the full history. The
    hourly pipeline must not call this -- see get_aqi_at below.
    """
    fg = get_feature_group()
    return _read_with_retry(fg.read, "Full feature-group read")


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

    Pushes the filter down to the feature store rather than reading everything
    and filtering in pandas. Downloading the whole group to look up one value
    is O(total rows) every hour: at 2k rows that was merely wasteful, but after
    backfilling to 49k it started dropping the Arrow Flight connection
    mid-transfer (FlightUnavailableError: Socket closed) and failed roughly a
    quarter of hourly runs.
    """
    fg = get_feature_group()
    query = fg.select(["timestamp", "aqi"]).filter(
        (fg.city_name == city_name) & (fg.timestamp == timestamp)
    )
    try:
        df = _read_with_retry(query.read, "Previous-hour AQI lookup")
    except RestAPIError as error:
        if _is_empty_feature_group(error):
            return None
        raise

    if df.empty:
        return None
    return df.sort_values("timestamp").iloc[-1]["aqi"]


def _is_empty_feature_group(error):
    """Hopsworks raises a generic RestAPIError when a feature group exists but
    has no materialised data yet, which is a normal state right after creating
    a new feature group version."""
    message = str(error).lower()
    return "no data" in message or "not found" in message or "empty" in message
