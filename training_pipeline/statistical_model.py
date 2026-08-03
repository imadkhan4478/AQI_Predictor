"""Classical statistical time-series model (ARIMA).

Unlike every other model here, this one ignores all weather/pollutant
features entirely and predicts future AQI purely from the AQI series'
own past values. It answers: how much of our predictive power is just
persistence/trend, versus actually needing the engineered features?
"""

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from training_pipeline.data import FORECAST_HORIZON_HOURS, load_raw_aqi_series

ARIMA_ORDER = (2, 1, 2)  # (autoregressive, differencing, moving-average) terms


def train_and_predict(train_timestamps, test_timestamps):
    full_series = load_raw_aqi_series()
    train_end = train_timestamps.max()
    train_series = full_series.loc[:train_end]

    # Every test row at time t needs a forecast at t + 72h, so forecast far
    # enough ahead to cover the furthest such target time.
    last_target_time = test_timestamps.max() + pd.Timedelta(hours=FORECAST_HORIZON_HOURS)
    steps_needed = int((last_target_time - train_end) / pd.Timedelta(hours=1))

    model = ARIMA(train_series, order=ARIMA_ORDER).fit()
    forecast = model.forecast(steps=steps_needed)

    target_times = test_timestamps + pd.Timedelta(hours=FORECAST_HORIZON_HOURS)
    return forecast.reindex(target_times).values
