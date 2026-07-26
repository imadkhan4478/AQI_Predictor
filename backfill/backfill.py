"""Backfill historical AQI feature rows into Hopsworks over a date range."""

import argparse
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
from dotenv import load_dotenv

from backfill.fetch_historical import get_pollution_history, get_weather_history
from feature_pipeline.aqi import compute_aqi
from feature_pipeline.hopsworks_store import get_feature_group

load_dotenv()


def build_rows(city_name, pollution_list, weather_hourly):
    weather_df = pd.DataFrame(weather_hourly)
    weather_df["timestamp"] = pd.to_datetime(weather_df["time"], utc=True)
    weather_df = weather_df.set_index("timestamp")

    rows = []
    for entry in pollution_list:
        timestamp = datetime.fromtimestamp(entry["dt"], tz=timezone.utc)
        if timestamp not in weather_df.index:
            continue
        weather = weather_df.loc[timestamp]
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
                "humidity": weather["relative_humidity_2m"],
                "pressure": int(round(weather["surface_pressure"])),
                "wind_speed": weather["wind_speed_10m"],
                "wind_deg": weather["wind_direction_10m"],
                "clouds": weather["cloud_cover"],
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
    previous_aqi = None
    for row in rows:
        row["aqi_change_rate"] = float(row["aqi"] - previous_aqi) if previous_aqi is not None else 0.0
        previous_aqi = row["aqi"]
    return rows


def run(start_date, end_date):
    api_key = os.environ["OPENWEATHER_API_KEY"]
    lat = os.environ["CITY_LAT"]
    lon = os.environ["CITY_LON"]
    city_name = os.environ["CITY_NAME"]

    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())

    pollution_list = get_pollution_history(lat, lon, api_key, start_ts, end_ts)
    weather_hourly = get_weather_history(lat, lon, start_date, end_date)
    rows = build_rows(city_name, pollution_list, weather_hourly)

    df = pd.DataFrame(rows)
    print(f"Built {len(df)} historical feature rows from {start_date} to {end_date}")

    fg = get_feature_group()
    fg.insert(df)
    print(f"Inserted {len(df)} rows into the feature store")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill historical AQI feature rows into Hopsworks.")
    default_end = datetime.now(timezone.utc).date()
    default_start = default_end - timedelta(days=30)
    parser.add_argument("--start-date", default=default_start.isoformat())
    parser.add_argument("--end-date", default=default_end.isoformat())
    args = parser.parse_args()
    run(args.start_date, args.end_date)
