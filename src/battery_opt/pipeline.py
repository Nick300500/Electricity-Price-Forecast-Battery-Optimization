"""End-to-end pipeline orchestration: load -> prepare -> train -> forecast -> dispatch -> signal report.

This is the terminal-driven counterpart to the notebook: same underlying
`battery_opt` functions, but run as a plain script instead of executing
notebook cells, so it can eventually be swapped for a persistent/rolling
retraining service without touching the notebook at all.
"""

import logging
import os

import pandas as pd

from . import backtest, data_loading, data_prep, modeling, smard_client
from .config import DATA_DIR, DEFAULT_STORAGE_PARAMS, RESULTS_DIR, ensure_dirs
from .optimizer import run_dispatch

logger = logging.getLogger(__name__)


def load_raw_data(data_dir=DATA_DIR):
    """Load and hourly-resample every raw source needed for the feature dataset."""
    logger.info("Loading raw data from %s", data_dir)

    smard_client.ensure_generation_forecast_data(data_dir)
    gen_forecast_2023 = data_loading.load_generation_forecast(smard_client.generation_forecast_path(data_dir, "2023"))
    gen_forecast_2024 = data_loading.load_generation_forecast(smard_client.generation_forecast_path(data_dir, "2024"))
    gen_forecast_2025 = data_loading.load_generation_forecast(smard_client.generation_forecast_path(data_dir, "2025"))
    gen_forecast_23_24_df = pd.concat([gen_forecast_2023, gen_forecast_2024], ignore_index=True)

    price_2024_df = data_loading.load_price_2024(os.path.join(data_dir, "price_data.csv"))
    price_2023_df = data_loading.load_price_2023(
        os.path.join(data_dir, "price_data_2023 and others non forecasted.csv")
    )
    price_co2_2025_df = data_loading.load_price_co2_2025(os.path.join(data_dir, "price_CO2_2025.csv"))

    gen_df_2023 = data_loading.load_generation(os.path.join(data_dir, "gen_2023.csv"))
    gen_df_2024 = data_loading.load_generation(os.path.join(data_dir, "gen_2024.csv"))
    gen_df_2025 = data_loading.load_generation(os.path.join(data_dir, "gen_2025.csv"))

    load_df_2023 = data_loading.load_load(os.path.join(data_dir, "load_2023.csv"))
    load_df_2024 = data_loading.load_load(os.path.join(data_dir, "load_2024.csv"))
    load_df_2025 = data_loading.load_load(os.path.join(data_dir, "load_2025.csv"))

    co2_auction_2024_df = data_loading.load_co2_auction_2024(os.path.join(data_dir, "CO2_2024.csv"))

    gas_price_2023 = data_loading.load_gas_price(os.path.join(data_dir, "gas_price_2023.csv"))
    gas_price_2024 = data_loading.load_gas_price(os.path.join(data_dir, "gas_price_2024.csv"))
    gas_price_2025 = data_loading.load_gas_price(os.path.join(data_dir, "gas_price_2025.csv"))

    coal_23_24 = data_loading.load_coal_price(os.path.join(data_dir, "coal_23_24.csv"))
    coal_25 = data_loading.load_coal_price(os.path.join(data_dir, "coal_25.csv"))

    gas_price_hourly_23_24_df = pd.concat(
        [data_loading.build_hourly_gas_price(gas_price_2023), data_loading.build_hourly_gas_price(gas_price_2024)],
        ignore_index=True,
    )
    gas_price_hourly_25_df = data_loading.build_hourly_gas_price(gas_price_2025)

    coal_price_hourly_24_df = data_loading.build_hourly_coal_price(coal_23_24, start_year=2023)
    coal_price_hourly_25_df = data_loading.build_hourly_coal_price(coal_25, start_year=2023)

    return dict(
        price_2023_df=price_2023_df,
        price_2024_df=price_2024_df,
        co2_auction_2024_df=co2_auction_2024_df,
        price_co2_2025_df=price_co2_2025_df,
        gen_df_2023=gen_df_2023,
        gen_df_2024=gen_df_2024,
        gen_df_2025=gen_df_2025,
        load_df_2023=load_df_2023,
        load_df_2024=load_df_2024,
        load_df_2025=load_df_2025,
        gas_price_hourly_23_24_df=gas_price_hourly_23_24_df,
        gas_price_hourly_25_df=gas_price_hourly_25_df,
        coal_price_hourly_24_df=coal_price_hourly_24_df,
        coal_price_hourly_25_df=coal_price_hourly_25_df,
        gen_forecast_23_24_df=gen_forecast_23_24_df,
        gen_forecast_25_df=gen_forecast_2025,
    )


def run(data_dir=DATA_DIR, results_dir=RESULTS_DIR, storage_params=None):
    """Run the full pipeline and write the Results/ output CSVs:
    - battery_signal_report_2025.csv (time, P_ch, P_dis, E, predicted_price, actual_price, profit)
    - battery_cumulative_profit_2025.csv (running profit total, sampled every 24h)

    Returns (signal_report, cumulative_profit) as DataFrames.
    """
    ensure_dirs()
    storage_params = storage_params or DEFAULT_STORAGE_PARAMS

    raw = load_raw_data(data_dir)

    logger.info("Building feature dataset")
    forecasting_data, testrun_df = data_prep.build_feature_dataset(**raw)

    logger.info("Training random forest")
    features, target = modeling.prepare_features_target(forecasting_data)
    X_train, X_val, y_train, y_val = modeling.train_val_split(features, target)
    rf_model, _, y_pred_val, metrics = modeling.train_random_forest(
        X_train, y_train, X_val, y_val
    )
    logger.info(
        "Validation MAE=%.2f EUR/MWh, RMSE=%.2f EUR/MWh", metrics["mae_val"], metrics["rmse_val"]
    )

    logger.info("Computing ground-truth 2025 dispatch (for evaluation)")
    testrun_indexed = testrun_df.set_index("time")
    reference_2025 = run_dispatch(testrun_indexed["price"], storage_params)
    reference_2025.index = pd.to_datetime(reference_2025.index, format="%d.%m.%Y %H:%M")
    theoretical_total_profit = reference_2025["profit"].sum() / 1000

    logger.info("Forecasting 2025 prices and dispatching the battery off the forecast")
    forecasted_prices_df = modeling.forecast_prices(rf_model, testrun_df, features.columns)
    forecasted_dispatch = run_dispatch(forecasted_prices_df["forecasted_price"], storage_params)

    signal_report = backtest.build_signal_report(
        forecasted_dispatch, forecasted_prices_df["forecasted_price"], reference_2025
    )
    output_path = os.path.join(results_dir, "battery_signal_report_2025.csv")
    signal_report.to_csv(output_path)

    cumulative_profit = backtest.cumulative_profit_by_day(signal_report)
    cumulative_output_path = os.path.join(results_dir, "battery_cumulative_profit_2025.csv")
    cumulative_profit.to_csv(cumulative_output_path)

    actual_generated_profit = signal_report["profit"].sum() / 1000
    logger.info("Theoretical (perfect-foresight) profit: %.2f EUR", theoretical_total_profit)
    logger.info("Actual (forecast-driven) profit: %.2f EUR", actual_generated_profit)
    logger.info("Wrote %s", output_path)
    logger.info("Wrote %s", cumulative_output_path)

    return signal_report, cumulative_profit
