"""Build the (features, future-AQI target) training set from the Feature Store."""

import numpy as np
import pandas as pd

from feature_pipeline.hopsworks_store import read_features_df

FORECAST_HORIZON_HOURS = 72

# Conditions observed at prediction time.
OBSERVED_COLUMNS = [
    "hour", "day_of_week",
    "temp", "feels_like", "humidity", "pressure", "wind_speed", "wind_deg", "clouds",
    "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3",
    "aqi", "aqi_change_rate",
]

# day-of-month and month are stored in the feature group but excluded here: with
# 5.7 years of data they measurably hurt (0.628 vs 0.645 at 24h). hour is kept,
# and encoded cyclically below so 23:00 and 00:00 are adjacent.
AQI_LAGS_HOURS = [1, 2, 3, 6, 12, 24]
AQI_ROLLING_WINDOWS = [3, 24, 72]

TARGET_COLUMN = "aqi_target"
DELTA_COLUMN = "aqi_delta"


def _lag_column_names():
    names = [f"aqi_lag_{lag}" for lag in AQI_LAGS_HOURS]
    for window in AQI_ROLLING_WINDOWS:
        names += [f"aqi_rmean_{window}", f"aqi_rstd_{window}"]
    return names


FEATURE_COLUMNS = OBSERVED_COLUMNS + _lag_column_names() + ["hour_sin", "hour_cos"]


def add_time_features(df):
    df = df.copy()
    df["day_of_week"] = df["timestamp"].dt.dayofweek  # 0=Monday .. 6=Sunday
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


def add_lag_features(df):
    """Lagged and rolling AQI, computed on a regular hourly grid.

    Every window is shifted by 1 first so a row never sees its own AQI in its own
    rolling statistics, and the grid is made continuous because ~5% of hours are
    missing -- shifting by row position across a gap mislabels the lag.
    """
    df = df.sort_values("timestamp").set_index("timestamp").asfreq("h")

    # Bridge short gaps only, never long outages. Numeric columns only: pandas
    # skips text columns here anyway, but passing them is deprecated.
    numeric_columns = df.select_dtypes(include="number").columns
    df[numeric_columns] = df[numeric_columns].interpolate(limit=6)

    for lag in AQI_LAGS_HOURS:
        df[f"aqi_lag_{lag}"] = df["aqi"].shift(lag)
    for window in AQI_ROLLING_WINDOWS:
        past = df["aqi"].shift(1)
        df[f"aqi_rmean_{window}"] = past.rolling(window).mean()
        df[f"aqi_rstd_{window}"] = past.rolling(window).std()

    return df.reset_index()


def add_target(df, horizon_hours=FORECAST_HORIZON_HOURS):
    """Attach the AQI value from exactly horizon_hours later, matched by actual
    timestamp (not row position) so gaps in the hourly data don't silently
    misalign the label. Also stores the change from now, which is what the models
    are trained on."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    aqi_by_time = df.set_index("timestamp")["aqi"]

    future_timestamps = df["timestamp"] + pd.Timedelta(hours=horizon_hours)
    df[TARGET_COLUMN] = aqi_by_time.reindex(future_timestamps).values
    df[DELTA_COLUMN] = df[TARGET_COLUMN] - df["aqi"]

    return df


def time_based_split(X, y, timestamps, horizon_hours=FORECAST_HORIZON_HOURS, test_fraction=0.2):
    """Split chronologically, then purge training rows whose *label* falls inside
    the test window.

    Chronological order alone is not enough: a row at T_split - 1h carries a
    label from T_split + 71h, so the model sees test-period AQI during training.
    At 72h that inflated R2 from 0.237 to 0.326.
    """
    order = timestamps.sort_values().index
    split_at = int(len(order) * (1 - test_fraction))
    train_idx, test_idx = order[:split_at], order[split_at:]

    test_start = timestamps.loc[test_idx].min()
    label_time = timestamps.loc[train_idx] + pd.Timedelta(hours=horizon_hours)
    purged_idx = train_idx[(label_time < test_start).values]

    dropped = len(train_idx) - len(purged_idx)
    if dropped:
        print(f"Purged {dropped} training rows whose {horizon_hours}h label falls inside the test window")

    return X.loc[purged_idx], X.loc[test_idx], y.loc[purged_idx], y.loc[test_idx]


def load_training_data(horizon_hours=FORECAST_HORIZON_HOURS, target=TARGET_COLUMN):
    """Returns (X, y, timestamps, current_aqi).

    `target` selects the absolute future AQI or the change from now. Models are
    trained on the change and anchored to the latest reading, which tracks the
    pollution level for free -- Lahore's mean AQI has fallen from ~387 (2020) to
    ~162 (2026), and tree models cannot extrapolate below what they trained on.
    """
    df = read_features_df()
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_target(df, horizon_hours)

    before = len(df)
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)
    print(f"Dropped {before - len(df)} rows with no {horizon_hours}h-ahead label or incomplete lags")

    return df[FEATURE_COLUMNS], df[target], df["timestamp"], df["aqi"]


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    X, y, timestamps, current_aqi = load_training_data()
    print(X.shape, y.shape)
    print(f"{len(FEATURE_COLUMNS)} features:", ", ".join(FEATURE_COLUMNS))
    print(y.describe())
