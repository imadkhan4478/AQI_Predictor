"""Compute US EPA Air Quality Index from pollutant concentrations."""

import math

# (concentration_low, concentration_high, aqi_low, aqi_high) per pollutant.
# PM2.5/PM10 in ug/m3; CO in ppm; O3, SO2, NO2 in ppb.
_BREAKPOINTS = {
    "pm2_5": [
        (0.0, 9.0, 0, 50),
        (9.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 125.4, 151, 200),
        (125.5, 225.4, 201, 300),
        (225.5, 325.4, 301, 500),
    ],
    "pm10": [
        (0, 54, 0, 50),
        (55, 154, 51, 100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 604, 301, 500),
    ],
    "o3": [
        (0, 54, 0, 50),
        (55, 70, 51, 100),
        (71, 85, 101, 150),
        (86, 105, 151, 200),
        (106, 200, 201, 300),
    ],
    "co": [
        (0.0, 4.4, 0, 50),
        (4.5, 9.4, 51, 100),
        (9.5, 12.4, 101, 150),
        (12.5, 15.4, 151, 200),
        (15.5, 30.4, 201, 300),
        (30.5, 50.4, 301, 500),
    ],
    "so2": [
        (0, 35, 0, 50),
        (36, 75, 51, 100),
        (76, 185, 101, 150),
        (186, 304, 151, 200),
        (305, 604, 201, 300),
        (605, 1004, 301, 500),
    ],
    "no2": [
        (0, 53, 0, 50),
        (54, 100, 51, 100),
        (101, 360, 101, 150),
        (361, 649, 151, 200),
        (650, 1249, 201, 300),
        (1250, 2049, 301, 500),
    ],
}

# Molecular weights (g/mol), for converting ug/m3 -> ppb/ppm at 25C/1atm.
_MOLECULAR_WEIGHTS = {"co": 28.01, "o3": 48.00, "so2": 64.066, "no2": 46.0055}

# EPA breakpoints assume concentrations truncated to this many decimal places
# before matching (e.g. PM10 to whole numbers) -- without this, values that
# fall in the gap between adjacent buckets (e.g. PM10 = 54.5) match nothing.
_TRUNCATE_DECIMALS = {"pm2_5": 1, "pm10": 0, "o3": 0, "co": 1, "so2": 0, "no2": 0}


def _ugm3_to_ppb(conc_ugm3, molecular_weight):
    return conc_ugm3 * 24.45 / molecular_weight


def _truncate(value, decimals):
    factor = 10**decimals
    return math.floor(value * factor) / factor


def _sub_index(pollutant, concentration):
    breakpoints = _BREAKPOINTS[pollutant]
    capped = _truncate(min(concentration, breakpoints[-1][1]), _TRUNCATE_DECIMALS[pollutant])
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= capped <= c_high:
            return round((i_high - i_low) / (c_high - c_low) * (capped - c_low) + i_low)
    return None


# (aqi_low, aqi_high, category_name, alert_severity, color). alert_severity is
# None for the two "safe" categories, otherwise a label the app can key alert
# styling/urgency off of. Colors are the official EPA/AirNow AQI colors --
# not our generic categorical palette. AQI color-coding is a globally
# recognized convention (like traffic lights), so matching the real-world
# standard beats forcing a brand-neutral scheme onto a domain that already
# has its own universally recognized color language.
_CATEGORIES = [
    (0, 50, "Good", None, "#00e400"),
    (51, 100, "Moderate", None, "#f2c200"),
    (101, 150, "Unhealthy for Sensitive Groups", "warning", "#ff7e00"),
    (151, 200, "Unhealthy", "serious", "#ff0000"),
    (201, 300, "Very Unhealthy", "critical", "#8f3f97"),
    (301, 500, "Hazardous", "critical", "#7e0023"),
]


def aqi_category(aqi_value):
    """EPA category name, alert severity ('warning'/'serious'/'critical'/None),
    and the official EPA color for an AQI value."""
    for low, high, name, severity, color in _CATEGORIES:
        if low <= aqi_value <= high:
            return name, severity, color
    return ("Hazardous", "critical", "#7e0023") if aqi_value > 500 else ("Good", None, "#00e400")


def compute_aqi(components):
    """components: dict with keys co, no2, o3, so2, pm2_5, pm10 in ug/m3 (as returned by OpenWeather)."""
    sub_indices = {}
    for pollutant, concentration in components.items():
        if pollutant not in _BREAKPOINTS:
            continue
        if pollutant in _MOLECULAR_WEIGHTS:
            concentration = _ugm3_to_ppb(concentration, _MOLECULAR_WEIGHTS[pollutant])
            if pollutant == "co":
                concentration /= 1000  # ppb -> ppm
        sub_indices[pollutant] = _sub_index(pollutant, concentration)

    overall_aqi = max(sub_indices.values())
    dominant_pollutant = max(sub_indices, key=sub_indices.get)
    return overall_aqi, dominant_pollutant, sub_indices


if __name__ == "__main__":
    sample = {"co": 342.12, "no": 0.14, "no2": 2.1, "o3": 111.46, "so2": 1.69, "pm2_5": 37.3, "pm10": 39.1, "nh3": 5.92}
    aqi, dominant, breakdown = compute_aqi(sample)
    print("AQI:", aqi, "| dominant pollutant:", dominant)
    print("breakdown:", breakdown)
