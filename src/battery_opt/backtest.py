"""Backtesting: perfect-foresight vs. forecast-driven profit, sensitivity analysis, payback.

Consolidates several near-identical blocks from the original notebook into
shared functions:
  * every "dispatch on a price series -> save to Results/" step
  * every "take a dispatch's P_ch/P_dis/E and re-price it against the actual
    Day-Ahead price" step (this is what turns a forecast-driven schedule into
    a real €-profit figure)
  * the ±1€/MWh and ±X% forecast-correction functions used for the
    sensitivity sweeps
"""

import os

import numpy as np
import pandas as pd

from .config import RESULTS_DIR
from .optimizer import run_dispatch


def load_dispatch_csv(path, date_format=None):
    """Read a saved dispatch/price CSV and parse its datetime index."""
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, format=date_format)
    return df


def run_and_save_dispatch(price_series, output_filename, storage_params=None, results_dir=RESULTS_DIR):
    """Run the optimizer on a price series and persist the dispatch to Results/."""
    results_df = run_dispatch(price_series, storage_params)
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, output_filename)
    results_df.to_csv(output_path)
    return results_df


def compute_actual_profit(dispatch_df, reference_price_df):
    """Re-price a (possibly forecast-driven) dispatch schedule against actual prices.

    ``dispatch_df`` supplies the charge/discharge/state-of-charge schedule
    (columns P_ch, P_dis, E) that the optimizer chose for some price series
    (perfect, forecasted, or a sensitivity-corrected variant). ``reference_price_df``
    supplies the ground-truth index and 'price' column (typically the
    perfect-foresight result) the schedule is actually settled against.
    """
    actual_df = dispatch_df[["P_ch", "P_dis", "E"]].copy()
    actual_df.index = reference_price_df.index
    actual_df["price"] = reference_price_df["price"]
    actual_df["profit"] = (
        actual_df["P_dis"] * actual_df["price"] - actual_df["P_ch"] * actual_df["price"]
    )
    return actual_df


def build_signal_report(dispatch_df, predicted_price, reference_price_df):
    """Full report of how the battery would actually be run off a forecast signal.

    Columns: time (index), P_ch, P_dis, E, predicted_price, actual_price, profit.
    ``dispatch_df`` is the schedule the optimizer chose for ``predicted_price``
    (the "signal"); ``reference_price_df`` supplies the real Day-Ahead price
    the schedule settles against, which is what ``profit`` is computed from.
    """
    report = dispatch_df[["P_ch", "P_dis", "E"]].copy()
    report.index = reference_price_df.index
    report.index.name = "time"
    report["predicted_price"] = predicted_price.values
    report["actual_price"] = reference_price_df["price"].values
    report["profit"] = (
        report["P_dis"] * report["actual_price"] - report["P_ch"] * report["actual_price"]
    )
    return report


def cumulative_profit_by_day(report_df, profit_col="profit"):
    """Running total of ``profit_col``, sampled at the end of each 24h period.

    ``report_df`` needs a (roughly hourly) datetime index. Returns a single
    'cumulative_profit' column indexed by day.
    """
    cumulative = report_df[profit_col].cumsum()
    daily = cumulative.resample("24h").last()
    daily.name = "cumulative_profit"
    return daily.to_frame()


def correct_price_toward_actual_abs(forecasted_price, real_price, step=1.0):
    """Move the forecast up to `step` units closer to the actual price (never overshoot)."""
    diff = real_price - forecasted_price
    moved = forecasted_price + np.sign(diff) * step
    return np.where(diff.abs() <= step, real_price, moved)


def correct_price_away_from_actual_abs(forecasted_price, real_price, step=1.0):
    """Move the forecast `step` units further from the actual price (unconditionally)."""
    return np.where(real_price > forecasted_price, forecasted_price - step, forecasted_price + step)


def correct_price_fraction(forecasted_price, real_price, fraction, toward_actual=True):
    """Move the forecast a `fraction` of the gap towards (or away from) the actual price.

    fraction=0.1 reproduces the notebook's "±10%" sensitivity correction;
    the "Extra" 10-90% sweep reuses this with fraction in {0.1, ..., 0.9}.
    """
    diff = real_price - forecasted_price
    sign = 1 if toward_actual else -1
    return forecasted_price + sign * diff * fraction


def payback_years(profit_half_year, cost_battery):
    """Percentage of the battery cost recovered in the observed period, and years to full payback."""
    percent_paid_back = profit_half_year / cost_battery
    years = 1 / percent_paid_back
    return percent_paid_back, years


def profit_percent_of_theoretical(profit, theoretical_profit):
    return profit / theoretical_profit * 100


def run_sensitivity_sweep(forecasted_price, real_price, fractions, toward_actual, storage_params=None):
    """Run the ±X% sensitivity sweep (the "Extra" section): for each fraction,
    correct the forecast, dispatch against it, then re-price against actual.

    Returns a dict {fraction: actual_profit_k_eur}.
    """
    profits = {}
    for fraction in fractions:
        corrected = correct_price_fraction(forecasted_price, real_price, fraction, toward_actual)
        corrected_series = pd.Series(corrected, index=forecasted_price.index)
        dispatch_df = run_dispatch(corrected_series, storage_params)
        reference_df = pd.DataFrame({"price": real_price})
        actual_df = compute_actual_profit(dispatch_df, reference_df)
        profits[fraction] = actual_df["profit"].sum() / 1000
    return profits
