"""Fetch historical weather (Open-Meteo) and air pollution (OpenWeather) data."""

import time

import requests

POLLUTION_HISTORY_URL = "http://api.openweathermap.org/data/2.5/air_pollution/history"
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

WEATHER_HOURLY_FIELDS = (
    "temperature_2m,apparent_temperature,relative_humidity_2m,"
    "surface_pressure,wind_speed_10m,wind_direction_10m,cloud_cover"
)

# OpenWeather's air-pollution archive starts here; requesting earlier just
# returns an empty list.
POLLUTION_HISTORY_START = "2020-11-27"

MAX_ATTEMPTS = 4


def _get_with_retry(url, params):
    """Multi-year backfills make hundreds of calls, so a single transient 5xx
    or dropped connection shouldn't throw away the whole run."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as error:
            if attempt == MAX_ATTEMPTS:
                raise
            wait_seconds = 2**attempt
            print(f"    request failed ({type(error).__name__}), retry {attempt}/{MAX_ATTEMPTS} in {wait_seconds}s")
            time.sleep(wait_seconds)


def get_pollution_history(lat, lon, api_key, start_ts, end_ts):
    payload = _get_with_retry(
        POLLUTION_HISTORY_URL,
        {"lat": lat, "lon": lon, "start": start_ts, "end": end_ts, "appid": api_key},
    )
    return payload.get("list", [])


def get_weather_history(lat, lon, start_date, end_date):
    payload = _get_with_retry(
        WEATHER_ARCHIVE_URL,
        {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": WEATHER_HOURLY_FIELDS,
            "wind_speed_unit": "ms",
            "timezone": "UTC",
        },
    )
    return payload["hourly"]
