"""Backfill historical AQI feature rows into Hopsworks over a date range.

Both source APIs cap how much history one request will return, so a multi-year
range is fetched in monthly chunks and inserted in batches. The original 90-day
backfill left the models badly data-starved: a learning curve over the 2,065
rows it produced was still climbing steeply at 100% of the data (24h-ahead R2
went 0.01 -> 0.34 between 75% and 100%), meaning the models were limited by
sample count rather than by the choice of algorithm.
"""

import argparse
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
from dotenv import load_dotenv
from requests.exceptions import RequestException

from backfill.fetch_historical import (
    POLLUTION_HISTORY_START,
    get_pollution_history,
    get_weather_history,
)
from feature_pipeline.aqi import compute_aqi
from feature_pipeline.hopsworks_store import get_feature_group

load_dotenv()

# Rows per Hopsworks insert. A single multi-year insert is a long-lived upload
# that tends to die on an unstable connection; batching keeps each write short
# and means a failure costs one batch, not the whole run.
INSERT_BATCH_ROWS = 5000
MAX_INSERT_ATTEMPTS = 3

# Courtesy pause between API calls (OpenWeather free tier allows 60/min).
REQUEST_PAUSE_SECONDS = 1.0

# The feature group's schema was locked in by the first backfill, so every
# insert must reproduce these exact types or Hopsworks rejects the batch.
INT_COLUMNS = ["hour", "day", "month", "humidity", "pressure", "wind_deg", "clouds", "aqi"]
FLOAT_COLUMNS = [
    "temp", "feels_like", "wind_speed", "co", "no", "no2",
    "o3", "so2", "pm2_5", "pm10", "nh3", "aqi_change_rate",
]


def month_chunks(start_date, end_date):
    """Split [start_date, end_date] into (start, end) pairs of at most one month."""
    chunks = []
    chunk_start = start_date
    while chunk_start <= end_date:
        next_month = (chunk_start.replace(day=1) + timedelta(days=32)).replace(day=1)
        chunk_end = min(next_month - timedelta(days=1), end_date)
        chunks.append((chunk_start, chunk_end))
        chunk_start = chunk_end + timedelta(days=1)
    return chunks


def fetch_range(lat, lon, api_key, start_date, end_date):
    """Fetch pollution + weather for the whole range, one month at a time."""
    chunks = month_chunks(start_date, end_date)
    pollution_list = []
    weather_frames = []

    for index, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        start_ts = int(datetime.combine(chunk_start, datetime.min.time(), tzinfo=timezone.utc).timestamp())
        end_ts = int(datetime.combine(chunk_end, datetime.max.time(), tzinfo=timezone.utc).timestamp())

        pollution_list.extend(get_pollution_history(lat, lon, api_key, start_ts, end_ts))
        weather_hourly = get_weather_history(lat, lon, chunk_start.isoformat(), chunk_end.isoformat())
        weather_frames.append(pd.DataFrame(weather_hourly))

        print(f"  [{index}/{len(chunks)}] {chunk_start} -> {chunk_end}: {len(pollution_list)} pollution records so far")
        time.sleep(REQUEST_PAUSE_SECONDS)

    weather_df = pd.concat(weather_frames, ignore_index=True)
    weather_df["timestamp"] = pd.to_datetime(weather_df["time"], utc=True)
    weather_df = weather_df.drop_duplicates(subset="timestamp").set_index("timestamp")
    return pollution_list, weather_df


def build_rows(city_name, pollution_list, weather_df):
    rows = []
    for entry in pollution_list:
        timestamp = datetime.fromtimestamp(entry["dt"], tz=timezone.utc)
        if timestamp not in weather_df.index:
            continue
        weather = weather_df.loc[timestamp]
        if weather.isnull().any():
            continue  # Open-Meteo returns nulls for the most recent few days
        components = entry["components"]
        aqi, dominant_pollutant, _ = compute_aqi(components)
        rows.append(
            {
                "city_name": city_name,
                "timestamp": timestamp,
                "hour": timestamp.hour,
                "day": timestamp.day,
                "month": timestamp.month,
                "temp": weather["temperature_2m"],
                "feels_like": weather["apparent_temperature"],
                "humidity": round(weather["relative_humidity_2m"]),
                "pressure": round(weather["surface_pressure"]),
                "wind_speed": weather["wind_speed_10m"],
                "wind_deg": round(weather["wind_direction_10m"]),
                "clouds": round(weather["cloud_cover"]),
                "co": components["co"],
                "no": components["no"],
                "no2": components["no2"],
                "o3": components["o3"],
                "so2": components["so2"],
                "pm2_5": components["pm2_5"],
                "pm10": components["pm10"],
                "nh3": components["nh3"],
                "dominant_pollutant": dominant_pollutant,
                "aqi": aqi,
            }
        )

    rows.sort(key=lambda r: r["timestamp"])

    # Change rate is only meaningful against the immediately preceding hour.
    # Measuring it against "whatever the previous row happened to be" would
    # make the feature mean something different across archive gaps than it
    # does in the live pipeline.
    previous_row = None
    for row in rows:
        if previous_row is not None and row["timestamp"] - previous_row["timestamp"] == timedelta(hours=1):
            row["aqi_change_rate"] = float(row["aqi"] - previous_row["aqi"])
        else:
            row["aqi_change_rate"] = 0.0
        previous_row = row
    return rows


def to_typed_frame(rows):
    """Cast to the feature group's locked schema. Without this, a batch whose
    values happen to all be whole numbers infers int64 for a 'double' column
    (or vice versa) and Hopsworks rejects the insert."""
    df = pd.DataFrame(rows)
    for column in INT_COLUMNS:
        df[column] = df[column].astype("int64")
    for column in FLOAT_COLUMNS:
        df[column] = df[column].astype("float64")
    return df


def insert_batched(fg, df):
    total_batches = (len(df) + INSERT_BATCH_ROWS - 1) // INSERT_BATCH_ROWS
    for batch_number, start in enumerate(range(0, len(df), INSERT_BATCH_ROWS), start=1):
        batch = df.iloc[start : start + INSERT_BATCH_ROWS]
        for attempt in range(1, MAX_INSERT_ATTEMPTS + 1):
            try:
                fg.insert(batch)
                break
            except RequestException as error:
                if attempt == MAX_INSERT_ATTEMPTS:
                    raise
                wait_seconds = attempt * 5
                print(f"    batch {batch_number} insert failed ({type(error).__name__}), retry in {wait_seconds}s")
                time.sleep(wait_seconds)
        print(f"  inserted batch {batch_number}/{total_batches} ({len(batch)} rows)")


def run(start_date, end_date):
    api_key = os.environ["OPENWEATHER_API_KEY"]
    lat = os.environ["CITY_LAT"]
    lon = os.environ["CITY_LON"]
    city_name = os.environ["CITY_NAME"]

    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    archive_start = datetime.strptime(POLLUTION_HISTORY_START, "%Y-%m-%d").date()
    if start < archive_start:
        print(f"OpenWeather pollution history starts {archive_start}; clamping start date")
        start = archive_start

    print(f"Fetching {city_name} from {start} to {end} in monthly chunks...")
    pollution_list, weather_df = fetch_range(lat, lon, api_key, start, end)

    rows = build_rows(city_name, pollution_list, weather_df)
    df = to_typed_frame(rows)
    print(f"Built {len(df)} feature rows spanning {df['timestamp'].min()} -> {df['timestamp'].max()}")

    fg = get_feature_group()
    insert_batched(fg, df)
    print(f"Inserted {len(df)} rows into the feature store")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill historical AQI feature rows into Hopsworks.")
    parser.add_argument("--start-date", default=POLLUTION_HISTORY_START)
    parser.add_argument("--end-date", default=datetime.now(timezone.utc).date().isoformat())
    args = parser.parse_args()
    run(args.start_date, args.end_date)
