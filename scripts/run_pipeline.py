#!/usr/bin/env python3
"""Terminal entry point for the battery price-forecast/dispatch pipeline.

    python scripts/run_pipeline.py

Interim replacement for running the notebook: same battery_opt package,
just orchestrated from the command line. This will become the basis for a
persistent/rolling-retrain service later.
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from battery_opt import pipeline  # noqa: E402
from battery_opt.config import DATA_DIR, RESULTS_DIR  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DATA_DIR, help=f"Directory with the raw source CSVs (default: {DATA_DIR})")
    parser.add_argument("--results-dir", default=RESULTS_DIR, help=f"Directory to write outputs to (default: {RESULTS_DIR})")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    pipeline.run(data_dir=args.data_dir, results_dir=args.results_dir)


if __name__ == "__main__":
    main()
