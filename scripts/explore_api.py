"""Throwaway script to inspect the raw shape of the OpenWeather APIs before building the real feature pipeline."""

import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["OPENWEATHER_API_KEY"]
LAT = os.environ["CITY_LAT"]
LON = os.environ["CITY_LON"]

WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
POLLUTION_URL = "https://api.openweathermap.org/data/2.5/air_pollution"


def fetch(url):
    response = requests.get(url, params={"lat": LAT, "lon": LON, "appid": API_KEY, "units": "metric"})
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    print("=== Current Weather ===")
    print(json.dumps(fetch(WEATHER_URL), indent=2))

    print("\n=== Air Pollution ===")
    print(json.dumps(fetch(POLLUTION_URL), indent=2))
