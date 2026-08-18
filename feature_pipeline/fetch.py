"""Fetch raw weather and air pollution data from OpenWeather."""

import requests

WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
POLLUTION_URL = "https://api.openweathermap.org/data/2.5/air_pollution"
POLLUTION_HISTORY_URL = "http://api.openweathermap.org/data/2.5/air_pollution/history"


def get_weather(lat, lon, api_key):
    response = requests.get(WEATHER_URL, params={"lat": lat, "lon": lon, "appid": api_key, "units": "metric"})
    response.raise_for_status()
    return response.json()


def get_pollution(lat, lon, api_key):
    response = requests.get(POLLUTION_URL, params={"lat": lat, "lon": lon, "appid": api_key})
    response.raise_for_status()
    return response.json()


def get_pollution_at(lat, lon, api_key, timestamp):
    """Pollutant components recorded at exactly `timestamp`, or None.

    Used to rebuild the previous hour's AQI without querying the feature store.
    The archive is keyed to whole hours, so `timestamp` must already be floored;
    a window either side of it is requested and the exact hour selected.
    """
    target = int(timestamp.timestamp())
    response = requests.get(
        POLLUTION_HISTORY_URL,
        params={"lat": lat, "lon": lon, "start": target - 1800, "end": target + 1800, "appid": api_key},
        timeout=30,
    )
    response.raise_for_status()
    for entry in response.json().get("list", []):
        if entry["dt"] == target:
            return entry["components"]
    return None
