"""Fetch raw weather and air pollution data from OpenWeather."""

import requests

WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
POLLUTION_URL = "https://api.openweathermap.org/data/2.5/air_pollution"


def get_weather(lat, lon, api_key):
    response = requests.get(WEATHER_URL, params={"lat": lat, "lon": lon, "appid": api_key, "units": "metric"})
    response.raise_for_status()
    return response.json()


def get_pollution(lat, lon, api_key):
    response = requests.get(POLLUTION_URL, params={"lat": lat, "lon": lon, "appid": api_key})
    response.raise_for_status()
    return response.json()
