"""Classical statistical baseline (ARIMA) over AQI's own past.

Unlike every other candidate this one sees no weather or pollutant features at
all, only the AQI series. It answers how much of the predictive power comes from
the engineered features rather than from the series' own autocorrelation.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from training_pipeline.data import load_raw_aqi_series

ARIMA_ORDER = (2, 1, 2)  # (autoregressive, differencing, moving-average) terms

# ARIMA parameters are local, and fitting (2,1,2) over 5.7 years of hourly data
# costs minutes for a fit that barely differs from one year.
TRAILING_FIT_HOURS = 8760

# Forecast origins are advanced in blocks rather than hour by hour. Every test
# row still gets a prediction; rows inside a block are served by the origin at
# its start, so a prediction can be up to this many hours stale.
ORIGIN_STRIDE_HOURS = 6


def predict_delta(context):
    """Predicted change in AQI over the horizon, from a rolling-origin ARIMA.

    Refitting at every origin is not affordable, but forecasting the whole test
    span from a single fit is not a forecast -- it converges to the series mean
    within a day. So the model is fitted once and then *extended* with observed
    values as the origin advances (`append(..., refit=False)`), which keeps the
    forecast anchored to recent data at the cost of frozen coefficients.
    """
    horizon_hours = context["horizon_hours"]
    test_timestamps = context["test_timestamps"]
    train_end = context["train_timestamps"].max()

    series = load_raw_aqi_series()
    fit_start = train_end - pd.Timedelta(hours=TRAILING_FIT_HOURS)
    fitted = ARIMA(series.loc[fit_start:train_end], order=ARIMA_ORDER).fit()

    # One forecast per origin, then mapped onto the test rows it covers.
    origins = pd.date_range(
        start=train_end,
        end=test_timestamps.max(),
        freq=f"{ORIGIN_STRIDE_HOURS}h",
    )
    predicted_at_origin = pd.Series(index=origins, dtype=float)

    for origin in origins:
        forecast = fitted.forecast(steps=horizon_hours)
        predicted_at_origin.loc[origin] = forecast.iloc[-1]

        next_origin = origin + pd.Timedelta(hours=ORIGIN_STRIDE_HOURS)
        observed = series.loc[origin + pd.Timedelta(hours=1) : next_origin]
        if observed.empty:
            break
        fitted = fitted.append(observed, refit=False)

    # Each test row takes the forecast from the most recent origin at or before it.
    predicted_absolute = predicted_at_origin.reindex(
        test_timestamps, method="ffill"
    ).to_numpy(dtype=float)

    # Returned as a delta so it is directly comparable with the feature-based
    # models, which all predict the change from the latest reading.
    return predicted_absolute - np.asarray(context["current_aqi_test"], dtype=float)
