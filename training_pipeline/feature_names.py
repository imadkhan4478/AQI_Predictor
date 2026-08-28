"""Human-readable names for the stored feature columns.

The dashboard's explainability panel used to print raw column names -- `aqi_rmean_72`,
`aqi_lag_1`, `hour_sin` -- which tell a reader nothing and read as unfinished work. The
mapping lives here rather than in the dashboard so the API, the report and any future
front end name a feature the same way.

Unmapped columns fall through to a readable default rather than raising: a new feature
should appear in a chart under a tolerable name, not break the panel.
"""

DISPLAY_NAMES = {
    "aqi": "Current AQI",
    "aqi_change_rate": "AQI change rate",
    "temp": "Temperature",
    "feels_like": "Apparent temperature",
    "humidity": "Humidity",
    "pressure": "Pressure",
    "wind_speed": "Wind speed",
    "wind_deg": "Wind direction",
    "clouds": "Cloud cover",
    "co": "Carbon monoxide (CO)",
    "no": "Nitric oxide (NO)",
    "no2": "Nitrogen dioxide (NO2)",
    "o3": "Ozone (O3)",
    "so2": "Sulfur dioxide (SO2)",
    "pm2_5": "PM2.5",
    "pm10": "PM10",
    "nh3": "Ammonia (NH3)",
    "hour": "Hour of day",
    "day_of_week": "Day of week",
    "hour_sin": "Hour of day (cyclic)",
    "hour_cos": "Hour of day (cyclic)",
}

for _lag in (1, 2, 3, 6, 12, 24):
    DISPLAY_NAMES[f"aqi_lag_{_lag}"] = f"AQI {_lag}h ago"

for _window in (3, 24, 72):
    DISPLAY_NAMES[f"aqi_rmean_{_window}"] = f"AQI {_window}h average"
    DISPLAY_NAMES[f"aqi_rstd_{_window}"] = f"AQI {_window}h variability"


def display_name(column):
    """A readable label for a stored column name."""
    if column in DISPLAY_NAMES:
        return DISPLAY_NAMES[column]
    return column.replace("_", " ").strip().capitalize()
