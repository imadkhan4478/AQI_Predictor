"""Register one forecast model per horizon (24h/48h/72h), so the dashboard can
show a real day-by-day 3-day forecast rather than a single far-off number.

This is the entry point the daily GitHub Actions job runs. Each horizon is
trained, walk-forward evaluated, given its own blend weight and registered
independently -- deliberately, because the earlier shortcut of running the
bake-off once at 72h and assuming the winner generalised to 24h and 48h chose
the wrong model for all three. See docs/EXPERIMENT_LOG.md, Phase 3-4.
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
