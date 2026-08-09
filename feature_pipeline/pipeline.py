"""Fetch weather + pollution data, compute AQI, and write one feature row to Hopsworks."""

import os
import time
from datetime import datetime, timezone

import pandas as pd
from requests.exceptions import ConnectionError as RequestsConnectionError
from dotenv import load_dotenv

from feature_pipeline.aqi import compute_aqi
from feature_pipeline.fetch import get_pollution, get_weather
from feature_pipeline.hopsworks_store import get_feature_group, get_latest_aqi

load_dotenv()

MAX_INSERT_ATTEMPTS = 3


def build_feature_row(city_name, weather, pollution, previous_aqi=None):
    components = pollution["list"][0]["components"]
    aqi, dominant_pollutant, _ = compute_aqi(components)
    timestamp = datetime.fromtimestamp(weather["dt"], tz=timezone.utc)

    return {
        "city_name": city_name,
        "timestamp": timestamp,
        "hour": timestamp.hour,
        "day": timestamp.day,
        "month": timestamp.month,
        # Explicit float() on every column the Feature Group expects as
        # 'double': the live pipeline inserts a single row at a time, so if a
        # pollutant reading happens to be a whole number (e.g. OpenWeather
        # returns "no": 0 instead of 0.0), pandas infers that column as int64
        # for this one-row DataFrame -- clashing with the 'double' schema
        # locked in during the original multi-row backfill and failing the
        # insert. Same root cause as the earlier "pressure" bug in backfill.py.
        "temp": float(weather["main"]["temp"]),
        "feels_like": float(weather["main"]["feels_like"]),
        "humidity": weather["main"]["humidity"],
        "pressure": weather["main"]["pressure"],
        "wind_speed": float(weather["wind"]["speed"]),
        "wind_deg": weather["wind"].get("deg", 0),
        "clouds": weather["clouds"]["all"],
        "co": float(components["co"]),
        "no": float(components["no"]),
        "no2": float(components["no2"]),
        "o3": float(components["o3"]),
        "so2": float(components["so2"]),
        "pm2_5": float(components["pm2_5"]),
        "pm10": float(components["pm10"]),
        "nh3": float(components["nh3"]),
        "dominant_pollutant": dominant_pollutant,
        "aqi": aqi,
        "aqi_change_rate": float(aqi - previous_aqi) if previous_aqi is not None else 0.0,
    }


def run():
    api_key = os.environ["OPENWEATHER_API_KEY"]
    lat = os.environ["CITY_LAT"]
    lon = os.environ["CITY_LON"]
    city_name = os.environ["CITY_NAME"]

    weather = get_weather(lat, lon, api_key)
    pollution = get_pollution(lat, lon, api_key)
    previous_aqi = get_latest_aqi(city_name)
    row = build_feature_row(city_name, weather, pollution, previous_aqi)
    df = pd.DataFrame([row])

    fg = get_feature_group()
    for attempt in range(1, MAX_INSERT_ATTEMPTS + 1):
        try:
            fg.insert(df)
            break
        except RequestsConnectionError:
            if attempt == MAX_INSERT_ATTEMPTS:
                raise
            wait_seconds = attempt * 2
            print(
                f"Hopsworks insert connection failed (attempt {attempt}/{MAX_INSERT_ATTEMPTS}). "
                f"Retrying in {wait_seconds}s..."
            )
            time.sleep(wait_seconds)
    print(f"Inserted feature row for {city_name} at {row['timestamp']} (AQI={row['aqi']})")


if __name__ == "__main__":
    run()
