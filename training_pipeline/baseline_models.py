"""Candidate models, plus the naive baseline every one of them must beat."""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42


def build_ridge():
    # Linear models are scale-sensitive (co ~hundreds vs wind_speed <5); the tree
    # models below split on raw thresholds and are not.
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0))


def build_random_forest():
    return RandomForestRegressor(
        n_estimators=200, min_samples_leaf=2, random_state=RANDOM_SEED, n_jobs=-1
    )


def build_gradient_boosting():
    """Winner under walk-forward evaluation at all three horizons (24h R2 0.780
    vs RandomForest 0.759, Ridge 0.735).

    The ranking is evaluation-dependent: under a single frozen split Ridge wins,
    because a stale model cannot track a falling trend and linear models
    extrapolate. With frequent retraining that advantage disappears."""
    return HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, l2_regularization=1.0, random_state=RANDOM_SEED
    )


def predict_persistence(current_aqi):
    """The naive forecast: assume AQI does not change.

    Deceptively strong, because pollution is highly autocorrelated -- R2 0.814 /
    0.709 / 0.628 at 24/48/72h, beating every standalone model tried. Scored
    alongside every candidate for that reason.
    """
    return np.asarray(current_aqi, dtype=float)
