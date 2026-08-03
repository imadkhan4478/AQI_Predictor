"""SHAP feature-importance explanation for the winning model (Random Forest)."""

import shap
from dotenv import load_dotenv
from matplotlib import pyplot as plt

from training_pipeline.baseline_models import build_random_forest
from training_pipeline.data import load_training_data, time_based_split

load_dotenv()

OUTPUT_PATH = "reports/shap_feature_importance.png"


def run():
    X, y, timestamps = load_training_data()
    X_train, X_test, y_train, y_test = time_based_split(X, y, timestamps)

    model = build_random_forest()
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    print(f"Saved SHAP feature-importance plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
