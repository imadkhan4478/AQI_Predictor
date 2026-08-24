"""SHAP feature-importance explanation for the deployed model.

Explains the model that is actually registered and served: HistGradientBoosting
predicting the *change* in AQI, evaluated per horizon. The values below are
therefore attributions on the delta, not on the AQI level -- a feature that
pushes the delta up is a feature that makes air quality worse over the horizon.

Earlier versions of this file explained a Random Forest predicting the absolute
AQI. That model is no longer what runs (see docs/EXPERIMENT_LOG.md, Phase 4), so
explaining it would have described a system nobody is using.
"""

import shap
from dotenv import load_dotenv
from matplotlib import pyplot as plt

from training_pipeline.baseline_models import build_gradient_boosting
from training_pipeline.data import (
    DELTA_COLUMN,
    FORECAST_HORIZON_HOURS,
    load_training_data,
    time_based_split,
)

load_dotenv()

OUTPUT_PATH = "reports/shap_feature_importance.png"

# SHAP on ~10k rows of a boosted ensemble is slow and adds nothing over a
# sample: feature *ranking* stabilises long before the last few thousand rows.
MAX_EXPLAINED_ROWS = 2000


def run(horizon_hours=FORECAST_HORIZON_HOURS, output_path=OUTPUT_PATH):
    X, y_delta, timestamps, _ = load_training_data(horizon_hours, target=DELTA_COLUMN)
    X_train, X_test, y_train, _ = time_based_split(X, y_delta, timestamps, horizon_hours)

    model = build_gradient_boosting()
    model.fit(X_train, y_train)

    explained = X_test.head(MAX_EXPLAINED_ROWS)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(explained)

    plt.figure()
    shap.summary_plot(shap_values, explained, plot_type="bar", show=False)
    plt.title(f"Drivers of the predicted {horizon_hours}h AQI change")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved SHAP feature-importance plot ({horizon_hours}h) to {output_path}")


if __name__ == "__main__":
    run()
