"""Candidate models, plus the naive baseline every one of them must beat."""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42


def build_ridge():
    # Ridge is a linear model, sensitive to feature scale (e.g. co ~hundreds
    # vs humidity 0-100 vs wind_speed <5), so it needs the features scaled
    # first. The tree models below don't -- they split on raw thresholds and
    # are scale-invariant.
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0))


def build_random_forest():
    return RandomForestRegressor(
        n_estimators=200, min_samples_leaf=2, random_state=RANDOM_SEED, n_jobs=-1
    )


def build_gradient_boosting():
    """Winner under walk-forward evaluation at all three horizons (24h R2 0.780
    vs RandomForest 0.759 and Ridge 0.735). Note the ranking is evaluation
    dependent: under a single frozen split Ridge came out on top, because a
    frozen model cannot track Lahore's falling pollution level and linear models
    extrapolate. Once retraining is frequent that advantage disappears."""
    return HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, l2_regularization=1.0, random_state=RANDOM_SEED
    )


def predict_persistence(current_aqi):
    """The naive forecast: assume AQI does not change.

    Deceptively strong for air quality -- pollution is highly autocorrelated and
    the same hour tomorrow resembles this hour today. It scores R2 0.814 / 0.709
    / 0.628 at 24/48/72h, beating every standalone model tried. Any model that
    cannot beat this is not earning its complexity, so it is scored alongside
    every candidate rather than mentioned once and forgotten.
    """
    return np.asarray(current_aqi, dtype=float)
