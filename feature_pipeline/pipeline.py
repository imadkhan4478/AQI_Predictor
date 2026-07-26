"""Fetch weather + pollution data, compute AQI, and write one feature row to Hopsworks."""

import os
import tempfile
from datetime import datetime, timezone

import hopsworks
import pandas as pd
from dotenv import load_dotenv

from feature_pipeline.aqi import compute_aqi
from feature_pipeline.fetch import get_pollution, get_weather

load_dotenv()

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1


def build_feature_row(city_name, weather, pollution):
    components = pollution["list"][0]["components"]
    aqi, dominant_pollutant, _ = compute_aqi(components)

    return {
        "city_name": city_name,
        "timestamp": datetime.fromtimestamp(weather["dt"], tz=timezone.utc),
        "temp": weather["main"]["temp"],
        "feels_like": weather["main"]["feels_like"],
        "humidity": weather["main"]["humidity"],
        "pressure": weather["main"]["pressure"],
        "wind_speed": weather["wind"]["speed"],
        "wind_deg": weather["wind"].get("deg", 0),
        "clouds": weather["clouds"]["all"],
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


def run():
    api_key = os.environ["OPENWEATHER_API_KEY"]
    lat = os.environ["CITY_LAT"]
    lon = os.environ["CITY_LON"]
    city_name = os.environ["CITY_NAME"]

    weather = get_weather(lat, lon, api_key)
    pollution = get_pollution(lat, lon, api_key)
    row = build_feature_row(city_name, weather, pollution)
    df = pd.DataFrame([row])

    project = hopsworks.login(
        project=os.environ["HOPSWORKS_PROJECT_NAME"],
        api_key_value=os.environ["HOPSWORKS_API_KEY"],
        cert_folder=os.path.join(tempfile.gettempdir(), "hopsworks_certs"),
    )
    fs = project.get_feature_store()
    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description="Hourly weather + pollutant readings with computed AQI",
        primary_key=["city_name", "timestamp"],
        event_time="timestamp",
        time_travel_format="HUDI",
    )
    fg.insert(df)
    print(f"Inserted feature row for {city_name} at {row['timestamp']} (AQI={row['aqi']})")


if __name__ == "__main__":
    run()
