"""Scikit-learn baseline models: Ridge Regression and Random Forest."""

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_ridge():
    # Ridge is a linear model, sensitive to feature scale (e.g. co ~hundreds
    # vs humidity 0-100 vs wind_speed <5), so it needs the features scaled
    # first. Random Forest below doesn't need this -- trees split on raw
    # thresholds and are scale-invariant.
    return make_pipeline(StandardScaler(), Ridge())


def build_random_forest():
    return RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
