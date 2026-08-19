"""Minimal client for the ENTSO-E Transparency Platform REST API.

Fetches the day-ahead total load forecast (document type A65, process type
A01) for the DE-LU bidding zone. This is the actual upstream source SMARD's
"Prognostizierter Stromverbrauch" republishes (per SMARD's own user manual)
— unlike SMARD's redesigned download-center frontend (undocumented, and its
backend schema couldn't be reverse-engineered), ENTSO-E's API is officially
documented and stable: https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html

Requires a free API token: register at https://transparency.entsoe.eu, then
email transparency@entsoe.eu with "RESTful API access" as the subject.
Approval typically takes a few working days. Once you have a token, set it
as the ENTSOE_API_TOKEN environment variable (don't hardcode it in source —
it's a credential).
"""

import logging
import os
import xml.etree.ElementTree as ET

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://web-api.tp.entsoe.eu/api"

# Germany/Luxembourg bidding zone (current since 2018-10-01); verified
# against the widely-used entsoe-py package's Area mapping.
DE_LU_BIDDING_ZONE = "10Y1001A1001A82H"

TOKEN_ENV_VAR = "ENTSOE_API_TOKEN"

# Matches the 2023 / 2024 / 2025 (H1) split the rest of Data/ uses.
LOAD_FORECAST_PERIODS = {
    "2023": ("2023-01-01", "2024-01-01"),
    "2024": ("2024-01-01", "2025-01-01"),
    "2025": ("2025-01-01", "2025-07-01"),
}


def _get_token(token=None):
    token = token or os.environ.get(TOKEN_ENV_VAR)
    if not token:
        raise RuntimeError(
            f"No ENTSO-E API token given. Set the {TOKEN_ENV_VAR} environment "
            "variable, or pass token= explicitly."
        )
    return token


def _local_tag(element):
    """Strip the XML namespace off an element's tag, e.g. '{ns}Foo' -> 'Foo'."""
    return element.tag.rsplit("}", 1)[-1]


def _parse_load_forecast_xml(xml_bytes):
    root = ET.fromstring(xml_bytes)

    if _local_tag(root) != "GL_MarketDocument":
        # Typically an Acknowledgement_MarketDocument carrying an error/empty-result reason.
        reason = root.find(".//{*}Reason/{*}text")
        message = reason.text if reason is not None else ET.tostring(root, encoding="unicode")[:500]
        raise RuntimeError(f"ENTSO-E API did not return load-forecast data: {message}")

    rows = []
    for period in root.findall(".//{*}TimeSeries/{*}Period"):
        start = pd.Timestamp(period.find("{*}timeInterval/{*}start").text)
        resolution = period.find("{*}resolution").text  # e.g. "PT15M", "PT60M"
        step = pd.Timedelta(resolution.replace("PT", "").replace("M", "min"))
        for point in period.findall("{*}Point"):
            position = int(point.find("{*}position").text)
            quantity = float(point.find("{*}quantity").text)
            rows.append((start + (position - 1) * step, quantity))

    df = pd.DataFrame(rows, columns=["time", "load_forecast_mw"])
    return df.sort_values("time").drop_duplicates(subset="time").reset_index(drop=True)


def fetch_load_forecast(start, end, bidding_zone=DE_LU_BIDDING_ZONE, token=None):
    """Fetch the day-ahead total load forecast for [start, end) at native (usually 15-min) resolution.

    Returns a DataFrame with columns ['time', 'load_forecast_mw'].
    """
    token = _get_token(token)
    params = {
        "securityToken": token,
        "documentType": "A65",
        "processType": "A01",
        "outBiddingZone_Domain": bidding_zone,
        "periodStart": pd.Timestamp(start).strftime("%Y%m%d%H%M"),
        "periodEnd": pd.Timestamp(end).strftime("%Y%m%d%H%M"),
    }
    resp = requests.get(BASE_URL, params=params, timeout=60)
    resp.raise_for_status()
    return _parse_load_forecast_xml(resp.content)


def fetch_load_forecast_hourly(start, end, bidding_zone=DE_LU_BIDDING_ZONE, token=None):
    """Fetch and resample to hourly resolution, converted to naive German local time.

    Mean MW over the hour == MWh energy for that hour, matching the [MWh]
    convention used elsewhere in battery_opt.data_prep (SMARD does the same
    conversion for its own published figures, per their user manual).

    ENTSO-E returns UTC timestamps; converted to Europe/Berlin and stripped of
    tz info here so this lines up with every other (naive, local-time) series
    in the pipeline — see the same conversion in data_loading.load_generation_forecast
    for why this matters (a real bug there before this was caught).
    """
    df = fetch_load_forecast(start, end, bidding_zone, token)
    df["time"] = df["time"].dt.tz_convert("Europe/Berlin").dt.tz_localize(None)
    hourly = df.set_index("time")["load_forecast_mw"].resample("1h").mean()
    return hourly.rename("net_load_forecast").reset_index()


def load_forecast_path(data_dir, year):
    return os.path.join(data_dir, f"load_forecast_{year}.csv")


def ensure_load_forecast_data(data_dir, force=False, token=None):
    """Download the load_forecast_<year>.csv files into data_dir if not already there.

    Safe/idempotent to call on every pipeline run — only hits the network for
    years whose file is missing (or all of them if force=True). Requires
    ENTSOE_API_TOKEN to be set (or token= passed) if anything actually needs
    fetching; does nothing if all files already exist.
    """
    os.makedirs(data_dir, exist_ok=True)

    for year, (start, end) in LOAD_FORECAST_PERIODS.items():
        out_path = load_forecast_path(data_dir, year)
        if os.path.exists(out_path) and not force:
            logger.debug("%s already exists, skipping", out_path)
            continue

        logger.info("Fetching day-ahead load forecast for %s (%s .. %s)", year, start, end)
        df = fetch_load_forecast_hourly(start, end, token=token)
        df.to_csv(out_path, index=False)
        logger.info("Wrote %s (%d rows)", out_path, len(df))
