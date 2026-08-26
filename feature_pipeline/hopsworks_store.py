"""Shared Hopsworks connection logic, used by both the live pipeline and backfill."""

import os
import tempfile
import time
from datetime import datetime, timedelta, timezone

import hopsworks
import pandas as pd
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
    """Arrow Flight drops connections intermittently on larger transfers. These
    are transient, so retry before failing the run."""
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
    """All rows in the feature group, as a DataFrame.

    For training and analysis only; the hourly pipeline must not call this.
    """
    fg = get_feature_group()
    return _read_with_retry(fg.read, "Full feature-group read")


SERVING_LOOKBACK_HOURS = 240


def read_recent_features_df(city_name, hours=SERVING_LOOKBACK_HOURS):
    """One city's rows over the last `hours`, for assembling a live feature row.

    Serving needs only enough history to fill the longest rolling window (72h) --
    every lag and rolling statistic is local to its window, so the newest row comes
    out identical to one computed over the full history. Reading all 49k rows over
    Arrow Flight instead was the slowest thing on the dashboard's cold path, and
    the transfer most likely to be dropped.
    """
    fg = get_feature_group()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    query = fg.filter((fg.city_name == city_name) & (fg.timestamp >= cutoff))
    return _read_with_retry(query.read, f"Recent feature read ({hours}h)")


OFFLINE_LOOKBACK_HOURS = 48


def offline_timestamps(city_name):
    """Every timestamp readable in the *offline* store for this city over the
    lookback window, as a Series. Empty if none.

    `fg.insert` writes through the online path and hands the offline table to a
    materialisation job. When that job stalls the insert still succeeds, while the
    offline table -- the only thing training and serving read -- stops advancing.
    Reading it back is the only way to assert on the state the insert was supposed
    to produce. Bounded to the last OFFLINE_LOOKBACK_HOURS so this stays a small
    transfer suitable for the hourly job.
    """
    fg = get_feature_group()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=OFFLINE_LOOKBACK_HOURS)
    query = fg.select(["timestamp"]).filter(
        (fg.city_name == city_name) & (fg.timestamp >= cutoff)
    )
    try:
        df = _read_with_retry(query.read, "Offline freshness check")
    except RestAPIError as error:
        if _is_empty_feature_group(error):
            return pd.Series([], dtype="datetime64[ns, UTC]")
        raise

    if df.empty:
        return pd.Series([], dtype="datetime64[ns, UTC]")
    return pd.to_datetime(df["timestamp"]).sort_values()


def get_aqi_at(city_name, timestamp):
    """AQI recorded for this city at exactly `timestamp`, or None if absent.

    Returns None for a missing hour rather than falling back to the latest row,
    so aqi_change_rate means the same thing in production as in the backfill.
    The narrow except matters: an empty feature group is expected on a fresh
    version, but an auth failure must not become "no previous reading".
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
    """Hopsworks raises a generic RestAPIError for a feature group that exists but
    has no materialised data yet -- normal for a fresh version."""
    message = str(error).lower()
    return "no data" in message or "not found" in message or "empty" in message
