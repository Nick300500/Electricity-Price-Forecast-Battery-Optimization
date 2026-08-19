"""Build the merged feature dataset used to train/evaluate the price forecast.

The training set (2023+2024, ``forecasting_data``) and the 2025 held-out set
(``testrun_df``) go through the exact same enrichment steps, so that logic
lives once in :func:`add_engineered_columns` instead of being duplicated per
dataset as in the original notebook.
"""

import pandas as pd

# net_load/res_load stay on realized ("Berechnete Auflösungen") values — no
# confirmed day-ahead load-forecast source yet (see
# smard_client.FORECAST_LOAD_FILTER). Generation is on forecasts (below).
GEN_LOAD_COLUMNS_TO_CONVERT = [
    "net_load [MWh]",
    "res_load [MWh]",
]

# battery_opt.smard_client.FORECAST_GENERATION_FILTERS key -> feature column name.
GENERATION_FORECAST_COLUMNS = {
    "total_forecast": "total_gen [MWh]",
    "solar_forecast": "solar_gen [MWh]",
    "wind_offshore_forecast": "wind_gen_off [MWh]",
    "wind_onshore_forecast": "wind_gen_on [MWh]",
    "other_forecast": "other_gen [MWh]",
}


def add_engineered_columns(df, gen_df, load_df, gas_price_hourly_df, coal_price_hourly_df, gen_forecast_df):
    """Attach generation-forecast/load/gas/coal columns to a price DataFrame (in place-ish, returns df).

    Assumes ``gen_df``/``load_df``/``gas_price_hourly_df``/``coal_price_hourly_df`` are
    already aligned by row order with ``df`` (hourly, same start date).
    ``gen_forecast_df`` (from data_loading.load_generation_forecast) is joined
    on its own 'time' column instead, since it comes from a different source
    (the SMARD API rather than a manually exported CSV) and isn't guaranteed
    to be in the same row order.
    """
    df["time"] = gen_df["Datum von"]
    df["net_load [MWh]"] = load_df["Netzlast [MWh] Berechnete Auflösungen"]
    df["res_load [MWh]"] = load_df["Residuallast [MWh] Berechnete Auflösungen"]
    df["gas_price [EUR/MWh]"] = gas_price_hourly_df["gas_price"]
    df["coal_price [EUR/t]"] = coal_price_hourly_df["hourly_coal_price"]

    for col in GEN_LOAD_COLUMNS_TO_CONVERT:
        df[col] = (
            df[col].astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        )
    df[GEN_LOAD_COLUMNS_TO_CONVERT] = df[GEN_LOAD_COLUMNS_TO_CONVERT].apply(pd.to_numeric, errors="coerce")

    time_key = pd.to_datetime(df["time"], format="%d.%m.%Y %H:%M")
    forecast_aligned = gen_forecast_df.set_index("time").reindex(time_key).reset_index(drop=True)
    for source_col, feature_col in GENERATION_FORECAST_COLUMNS.items():
        df[feature_col] = forecast_aligned[source_col].to_numpy()

    df["diff_load_gen [MWh]"] = df["total_gen [MWh]"] - df["net_load [MWh]"]

    # Historical day-ahead prices as lag features.
    # ATTENTION: until the first repetition occurs, no value is given (NaN).
    df["DA_price_1_day_before"] = df["price"].shift(24)
    df["DA_price_2_days_before"] = df["price"].shift(48)
    df["DA_price_1_week_before"] = df["price"].shift(168)

    return df


def add_calendar_columns(df, start_date, n_hours):
    """Add weekday/month (from a synthetic hourly date range) and hour-of-day (from 'time')."""
    date_range = pd.date_range(start=start_date, periods=n_hours, freq="h")
    df["weekday"] = date_range.dayofweek  # Monday=0, Sunday=6
    df["month"] = date_range.month

    df["hour"] = [t[-5:-3] for t in df["time"]]
    df["hour"] = df["hour"].astype(float)

    return df


def build_feature_dataset(
    price_2023_df,
    price_2024_df,
    co2_auction_2024_df,
    price_co2_2025_df,
    gen_df_2023,
    gen_df_2024,
    gen_df_2025,
    load_df_2023,
    load_df_2024,
    load_df_2025,
    gas_price_hourly_23_24_df,
    gas_price_hourly_25_df,
    coal_price_hourly_24_df,
    coal_price_hourly_25_df,
    gen_forecast_23_24_df,
    gen_forecast_25_df,
    start_date="2023-01-01",
    n_hours_train=17544,  # 2 years
    n_hours_test=4343,
):
    """Merge all raw sources into the training set and the 2025 held-out set.

    Returns
    -------
    (forecasting_data, testrun_df) : tuple of DataFrame
    """
    price_2024_df = price_2024_df.copy()
    price_2024_df["CO2 Emission Allowances, Auction DE"] = co2_auction_2024_df[
        "CO2 Emission Allowances, Auction DE"
    ]
    price_2024_df["CO2 Emission Allowances, Auction EU"] = co2_auction_2024_df[
        "CO2 Emission Allowances, Auction EU"
    ]

    forecasting_data = pd.concat([price_2023_df, price_2024_df], ignore_index=True)
    for col in ("Date (GMT+1)", "Time"):
        if col in forecasting_data.columns:
            forecasting_data = forecasting_data.drop(columns=[col])

    testrun_df = price_co2_2025_df.copy()

    gen_df = pd.concat([gen_df_2023, gen_df_2024], ignore_index=True).drop(columns=["Datum bis"])
    gen_df_2025_test = gen_df_2025.drop(columns=["Datum bis"])

    load_df = pd.concat([load_df_2023, load_df_2024], ignore_index=True).drop(columns=["Datum bis"])
    load_df_2025_test = load_df_2025.drop(columns=["Datum bis"])

    forecasting_data = add_engineered_columns(
        forecasting_data, gen_df, load_df, gas_price_hourly_23_24_df, coal_price_hourly_24_df,
        gen_forecast_23_24_df,
    )
    testrun_df = add_engineered_columns(
        testrun_df, gen_df_2025_test, load_df_2025_test, gas_price_hourly_25_df, coal_price_hourly_25_df,
        gen_forecast_25_df,
    )

    forecasting_data = add_calendar_columns(forecasting_data, start_date, n_hours_train)
    testrun_df = add_calendar_columns(testrun_df, start_date, n_hours_test)

    return forecasting_data, testrun_df
