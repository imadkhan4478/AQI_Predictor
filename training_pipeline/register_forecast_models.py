"""Register one forecast model per horizon (24h/48h/72h).

Entry point for the daily GitHub Actions job. Each horizon is trained, evaluated
and weighted independently: running the bake-off once at 72h and assuming the
winner generalised chose the wrong model for all three.
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
