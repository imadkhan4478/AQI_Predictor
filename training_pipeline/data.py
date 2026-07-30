"""Build the (features, 3-day-ahead AQI target) training set from the Feature Store."""

import pandas as pd

from feature_pipeline.hopsworks_store import read_features_df

FORECAST_HORIZON_HOURS = 72

FEATURE_COLUMNS = [
    "hour", "day", "month", "day_of_week",
    "temp", "feels_like", "humidity", "pressure", "wind_speed", "wind_deg", "clouds",
    "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3",
    "aqi", "aqi_change_rate",
]
TARGET_COLUMN = "aqi_target"


def add_target(df):
    """Attach the AQI value from exactly FORECAST_HORIZON_HOURS later, matched by
    actual timestamp (not row position) so gaps in the hourly data don't silently
    misalign the label."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    aqi_by_time = df.set_index("timestamp")["aqi"]

    future_timestamps = df["timestamp"] + pd.Timedelta(hours=FORECAST_HORIZON_HOURS)
    df[TARGET_COLUMN] = aqi_by_time.reindex(future_timestamps).values

    return df


def add_time_features(df):
    df["day_of_week"] = df["timestamp"].dt.dayofweek  # 0=Monday .. 6=Sunday
    return df


def load_training_data():
    df = read_features_df()
    df = add_time_features(df)
    df = add_target(df)

    before = len(df)
    df = df.dropna(subset=[TARGET_COLUMN])
    print(f"Dropped {before - len(df)} rows with no 3-day-ahead label available (gaps / end of data)")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    timestamps = df["timestamp"]
    return X, y, timestamps


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    X, y, timestamps = load_training_data()
    print(X.shape, y.shape)
    print(X.head())
    print(y.describe())
