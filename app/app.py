"""Streamlit dashboard: current AQI + 3-day forecast, with hazardous-AQI alerts."""

import os

import matplotlib.pyplot as plt
import streamlit as st
from dotenv import load_dotenv

from app.load_model import load_latest_model
from feature_pipeline.aqi import aqi_category
from feature_pipeline.hopsworks_store import read_features_df
from training_pipeline.data import FEATURE_COLUMNS, add_time_features

load_dotenv()

HORIZONS_HOURS = [24, 48, 72]
SEVERITY_RANK = {None: 0, "warning": 1, "serious": 2, "critical": 3}

# Fixed status colors -- never reused for anything else, always paired with a
# text label so severity is never communicated by color alone.
STATUS_COLOR = {
    None: "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}
STATUS_MESSAGE = {
    "warning": "Sensitive groups (children, elderly, respiratory/heart conditions) should limit prolonged outdoor exertion.",
    "serious": "Everyone may begin to experience health effects. Limit prolonged outdoor exertion.",
    "critical": "Health alert: everyone may experience serious health effects. Avoid outdoor activity.",
}


@st.cache_data(ttl=600)
def load_latest_row(city_name):
    df = read_features_df()
    df = add_time_features(df)
    city_rows = df[df["city_name"] == city_name].sort_values("timestamp")
    return city_rows.iloc[-1]


@st.cache_resource
def load_models():
    models = {}
    for horizon_hours in HORIZONS_HOURS:
        model, version, metrics = load_latest_model(f"aqi_random_forest_{horizon_hours}h")
        models[horizon_hours] = {"model": model, "version": version, "metrics": metrics}
    return models


def plot_forecast(forecast):
    labels = ["Now" if h == 0 else f"+{h}h" for h in forecast]
    values = list(forecast.values())
    categories = [aqi_category(v) for v in values]
    colors = [STATUS_COLOR[severity] for _, severity in categories]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=colors, width=0.5)
    for bar, value, (name, _) in zip(bars, values, categories):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 3, str(value), ha="center", fontweight="bold")
        ax.text(bar.get_x() + bar.get_width() / 2, 5, name, ha="center", rotation=90, color="white", fontsize=8)
    ax.set_ylabel("AQI")
    ax.set_title("3-Day AQI Forecast")
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def main():
    st.set_page_config(page_title="AQI Predictor", page_icon="🌫️")
    city_name = os.environ["CITY_NAME"]
    st.title(f"AQI Predictor — {city_name}")

    with st.spinner("Loading latest data and models..."):
        latest_row = load_latest_row(city_name)
        models = load_models()

    X_live = latest_row[FEATURE_COLUMNS].to_frame().T

    forecast = {0: int(latest_row["aqi"])}
    for horizon_hours in HORIZONS_HOURS:
        prediction = models[horizon_hours]["model"].predict(X_live)[0]
        forecast[horizon_hours] = round(prediction)

    current_category, _ = aqi_category(forecast[0])
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Current AQI", forecast[0], current_category)
    with col2:
        st.caption(f"As of {latest_row['timestamp']} UTC · dominant pollutant: {latest_row['dominant_pollutant']}")

    worst_horizon, worst_severity = max(
        ((h, aqi_category(v)[1]) for h, v in forecast.items()),
        key=lambda item: SEVERITY_RANK[item[1]],
    )
    if worst_severity:
        when = "now" if worst_horizon == 0 else f"in {worst_horizon}h"
        category_name, _ = aqi_category(forecast[worst_horizon])
        alert_fn = st.error if worst_severity == "critical" else st.warning
        alert_fn(f"**{category_name} air quality expected {when}** (AQI {forecast[worst_horizon]}). {STATUS_MESSAGE[worst_severity]}")

    st.pyplot(plot_forecast(forecast))

    with st.expander("Why does the model predict this? (SHAP feature importance)"):
        st.image("reports/shap_feature_importance.png")
        st.caption(
            "Feature importance for the Random Forest model, from the 4-model comparison "
            "(Ridge/RandomForest/NeuralNet/ARIMA) documented in training_pipeline/."
        )

    with st.expander("Model details"):
        for horizon_hours in HORIZONS_HOURS:
            info = models[horizon_hours]
            st.write(f"**{horizon_hours}h model** — v{info['version']} — {info['metrics']}")


if __name__ == "__main__":
    main()
