#!/usr/bin/env python3
"""Download SMARD day-ahead generation-forecast CSVs into Data/, if not already there.

Fixes the "actuals instead of forecasts" feature-realism issue from the
notebook review: wind/solar generation features were pulled from realized
values, which aren't available at bidding time in live operation. SMARD
publishes genuine day-ahead forecasts for these under separate filter ids
(see battery_opt.smard_client), confirmed against
https://github.com/bundesAPI/smard-api and by live probing.

pipeline.py calls the same underlying function automatically before every
run, so this script is only needed to pre-fetch the data on its own (e.g. to
inspect it, or to warm the cache before an offline run):

    python scripts/fetch_forecast_data.py
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from battery_opt import smard_client  # noqa: E402
from battery_opt.config import DATA_DIR  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--force", action="store_true", help="Redownload even if files already exist")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    smard_client.ensure_generation_forecast_data(args.data_dir, args.force)


if __name__ == "__main__":
    main()
