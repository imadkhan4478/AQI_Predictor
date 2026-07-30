"""Train and evaluate all candidate models, report which one wins."""

from dotenv import load_dotenv

from training_pipeline.baseline_models import build_random_forest, build_ridge
from training_pipeline.data import load_training_data, time_based_split
from training_pipeline.evaluate import evaluate

load_dotenv()


def run():
    X, y, timestamps = load_training_data()
    X_train, X_test, y_train, y_test = time_based_split(X, y, timestamps)
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows (most recent, held out)")

    candidates = {
        "Ridge": build_ridge(),
        "RandomForest": build_random_forest(),
    }

    results = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = evaluate(y_test, y_pred)

    print("\nModel comparison (on held-out, most-recent 20% of data):")
    for name, metrics in results.items():
        print(f"  {name:15s} RMSE={metrics['RMSE']:.2f}  MAE={metrics['MAE']:.2f}  R2={metrics['R2']:.3f}")

    return results


if __name__ == "__main__":
    run()
