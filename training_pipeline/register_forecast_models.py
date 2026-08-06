"""Register a Random Forest model for each forecast horizon (24h/48h/72h),
so the dashboard can show a real day-by-day 3-day forecast rather than a
single far-off number. Random Forest was already established as the best
model type via the 4-way comparison at the 72h horizon (train.py); we reuse
that same architecture per horizon rather than re-running the full bake-off
three times.
"""

from dotenv import load_dotenv

from training_pipeline.register import run as register_one_horizon

load_dotenv()

HORIZONS_HOURS = [24, 48, 72]


def run():
    for horizon_hours in HORIZONS_HOURS:
        print(f"\n=== Training + registering the {horizon_hours}h-ahead model ===")
        metrics = register_one_horizon(horizon_hours)
        print(f"{horizon_hours}h model metrics: {metrics}")


if __name__ == "__main__":
    run()
