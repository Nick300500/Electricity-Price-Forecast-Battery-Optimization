"""Minimal client for SMARD.de's public chart_data JSON API.

Endpoint structure and filter ids confirmed against
https://github.com/bundesAPI/smard-api (openapi.yaml) and by live probing:

    index:  {BASE_URL}/{filter}/{region}/index_{resolution}.json
            -> {"timestamps": [ms, ms, ...]}   (chunk start times, ~7 days apart for "hour")
    series: {BASE_URL}/{filter}/{region}/{filter}_{region}_{resolution}_{timestamp}.json
            -> {"series": [[ms, value], ...]}

Used to fetch day-ahead *forecast* generation, as a fix for the notebook
review's "actuals instead of forecasts" feature-realism issue: SMARD
publishes genuine day-ahead forecasts for wind/solar/other generation under
separate filter ids from the realized ("Berechnete Auflösungen") values the
rest of this project's Data/ CSVs use.
"""

import logging
import os

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://www.smard.de/app/chart_data"

# Matches the 2023 / 2024 / 2025 (H1) split the rest of Data/ uses.
GENERATION_FORECAST_PERIODS = {
    "2023": ("2023-01-01", "2024-01-01"),
    "2024": ("2024-01-01", "2025-01-01"),
    "2025": ("2025-01-01", "2025-07-01"),
}

# "Prognostizierte Erzeugung: ..." (day-ahead generation forecast) filter ids.
# Verified empirically: solar_forecast follows the expected 0-at-night /
# peak-at-noon curve, and wind_onshore + wind_offshore + solar + other sums
# exactly to total_forecast for spot-checked hours.
#
# NOTE: the community-maintained openapi.yaml this was checked against
# (bundesAPI/smard-api) has an internal inconsistency for the solar filter —
# its enum list contains 126, but its own description text maps
# "Prognostizierte Erzeugung: Photovoltaik" to 125. 126 returns a series with
# no day/night pattern at all (not solar); 125 is the one that matches actual
# solar behavior, confirmed above.
FORECAST_GENERATION_FILTERS = {
    "wind_onshore_forecast": 123,
    "wind_offshore_forecast": 3791,
    "solar_forecast": 125,
    "other_forecast": 715,
    "total_forecast": 122,
}

# No day-ahead *load* forecast filter id ("Prognostizierter Stromverbrauch")
# could be confirmed via the public API docs (bundesAPI/smard-api) — SMARD's
# website lists the category, but its filter id isn't in that spec. To fill
# this in: open SMARD's download center, select "Prognostizierter
# Stromverbrauch", start a manual download, and read the id out of the
# request URL in your browser's network tab (.../chart_data/<id>/DE/...).
FORECAST_LOAD_FILTER = None


def fetch_index(filter_id, region="DE", resolution="hour"):
    url = f"{BASE_URL}/{filter_id}/{region}/index_{resolution}.json"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()["timestamps"]


def fetch_chunk(filter_id, timestamp, region="DE", resolution="hour"):
    url = f"{BASE_URL}/{filter_id}/{region}/{filter_id}_{region}_{resolution}_{timestamp}.json"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()["series"]


def fetch_range(filter_id, start, end, region="DE", resolution="hour", value_col="value"):
    """Fetch [start, end) as a DataFrame with columns ['time', value_col].

    ``start``/``end`` are anything pandas.Timestamp accepts. Chunks are
    fetched from the last index timestamp at-or-before `start` through the
    last one before `end`, so partial chunks at the boundaries are included
    and then trimmed to the exact range.
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    timestamps = sorted(fetch_index(filter_id, region, resolution))
    before_start = [t for t in timestamps if t <= start_ts.timestamp() * 1000]
    within_range = [t for t in timestamps if start_ts.timestamp() * 1000 < t < end_ts.timestamp() * 1000]
    chunk_timestamps = ([before_start[-1]] if before_start else []) + within_range

    rows = []
    for ts in chunk_timestamps:
        rows.extend(fetch_chunk(filter_id, ts, region, resolution))

    df = pd.DataFrame(rows, columns=["timestamp_ms", value_col])
    df = df.dropna(subset=[value_col]).drop_duplicates(subset="timestamp_ms")
    df["time"] = pd.to_datetime(df["timestamp_ms"], unit="ms")
    df = df[(df["time"] >= start_ts) & (df["time"] < end_ts)]
    df = df.sort_values("time").drop(columns="timestamp_ms").reset_index(drop=True)
    return df[["time", value_col]]


def fetch_generation_forecast(start, end, region="DE", resolution="hour"):
    """Fetch all FORECAST_GENERATION_FILTERS series for [start, end) and merge on 'time'."""
    merged = None
    for name, filter_id in FORECAST_GENERATION_FILTERS.items():
        logger.info("Fetching %s (filter %s) for %s..%s", name, filter_id, start, end)
        df = fetch_range(filter_id, start, end, region, resolution, value_col=name)
        merged = df if merged is None else merged.merge(df, on="time", how="outer")
    return merged.sort_values("time").reset_index(drop=True)


def generation_forecast_path(data_dir, year):
    return os.path.join(data_dir, f"gen_forecast_{year}.csv")


def ensure_generation_forecast_data(data_dir, force=False):
    """Download the gen_forecast_<year>.csv files into data_dir if not already there.

    Safe/idempotent to call on every pipeline run — only hits the network for
    years whose file is missing (or all of them if force=True).
    """
    os.makedirs(data_dir, exist_ok=True)

    for year, (start, end) in GENERATION_FORECAST_PERIODS.items():
        out_path = generation_forecast_path(data_dir, year)
        if os.path.exists(out_path) and not force:
            logger.debug("%s already exists, skipping", out_path)
            continue

        logger.info("Fetching generation forecast for %s (%s .. %s)", year, start, end)
        df = fetch_generation_forecast(start, end)
        df.to_csv(out_path, index=False)
        logger.info("Wrote %s (%d rows)", out_path, len(df))

    if FORECAST_LOAD_FILTER is None:
        logger.warning(
            "No confirmed SMARD filter id for the day-ahead LOAD forecast "
            "('Prognostizierter Stromverbrauch') — net_load stays on realized "
            "values for now. See the comment on FORECAST_LOAD_FILTER for how "
            "to fill it in."
        )
