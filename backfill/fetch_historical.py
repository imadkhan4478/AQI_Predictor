"""Fetch historical weather (Open-Meteo) and air pollution (OpenWeather) data."""

import requests

POLLUTION_HISTORY_URL = "http://api.openweathermap.org/data/2.5/air_pollution/history"
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

WEATHER_HOURLY_FIELDS = (
    "temperature_2m,apparent_temperature,relative_humidity_2m,"
    "surface_pressure,wind_speed_10m,wind_direction_10m,cloud_cover"
)


def get_pollution_history(lat, lon, api_key, start_ts, end_ts):
    response = requests.get(
        POLLUTION_HISTORY_URL,
        params={"lat": lat, "lon": lon, "start": start_ts, "end": end_ts, "appid": api_key},
    )
    response.raise_for_status()
    return response.json()["list"]


def get_weather_history(lat, lon, start_date, end_date):
    response = requests.get(
        WEATHER_ARCHIVE_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": WEATHER_HOURLY_FIELDS,
            "wind_speed_unit": "ms",
            "timezone": "UTC",
        },
    )
    response.raise_for_status()
    return response.json()["hourly"]
