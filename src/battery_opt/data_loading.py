"""Raw-data loaders.

Each source (Day-Ahead price, CO2 auctions, generation, load, gas, coal)
ships as a differently-formatted CSV per year. These functions isolate the
per-source parsing quirks so the merge step (`data_prep.build_feature_dataset`)
can work with already-cleaned, comparable frames.
"""

import pandas as pd


def load_price_2024(path="price_data.csv"):
    """Day-Ahead price (+ already-named 'price' column) for 2024."""
    return pd.read_csv(path)


def load_price_2023(path="price_data_2023 and others non forecasted.csv"):
    """Day-Ahead price and misc. columns for 2023 (different export format)."""
    df = pd.read_csv(path, skiprows=[1])
    df = df.drop(columns=["Date (GMT+1)"])
    df = df.drop(
        columns=[
            "Hydro pumped storage consumption",
            "Cross border electricity trading",
            "Nuclear",
            "Non-Renewable",
            "Renewable",
        ]
    )
    df = df.rename(columns={"Day Ahead Auction (DE-LU)": "price"})
    return df


def load_price_co2_2025(path="price_CO2_2025.csv", n_rows=4343):
    """Day-Ahead price + CO2 auction prices for 2025 (until end of June)."""
    df = pd.read_csv(path, delimiter=",", skiprows=[1])
    df = df.drop(columns=["Datum (MEZ)"])
    df = df.rename(
        columns={
            "Day Ahead Auktion (DE-LU)": "price",
            "CO2 Emissionszertifikate, Auktion DE": "CO2 Emission Allowances, Auction DE",
            "CO2 Emissionszertifikate, Auktion EU": "CO2 Emission Allowances, Auction EU",
        }
    )
    return df.iloc[0:n_rows]


def load_co2_auction_2024(path="CO2_2024.csv"):
    return pd.read_csv(path, delimiter=",", skiprows=[1])


def load_generation(path, delimiter=","):
    return pd.read_csv(path, delimiter=delimiter)


def load_generation_forecast(path):
    """Load a SMARD day-ahead generation-forecast CSV (see smard_client.py /
    scripts/fetch_forecast_data.py). Already-clean floats, unlike the manually
    exported actuals CSVs — but its 'time' column decodes from SMARD's epoch-ms
    timestamps as naive datetimes that are actually UTC instants, not German
    local time like every other timestamp in this pipeline ("Datum von" in the
    manually exported CSVs is local). Convert explicitly so joins against the
    rest of the feature set line up correctly across DST changes (verified by
    cross-checking SMARD's actual-generation series against the manually
    exported actuals CSV for the same day: values match at a UTC+2 shift in
    summer, not at zero offset)."""
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("Europe/Berlin").dt.tz_localize(None)
    # DST fall-back duplicates one local wall-clock hour (e.g. 2023-10-29 02:00
    # occurs twice in UTC-derived data); keep the first to make 'time' unique
    # for the reindex in add_engineered_columns. DST spring-forward's missing
    # hour is left as-is -- it naturally becomes a NaN row on reindex, which
    # the rest of this pipeline already tolerates (e.g. the lag-feature NaNs
    # at the start of the series).
    df = df.drop_duplicates(subset="time", keep="first")
    return df


def load_load(path, delimiter=";"):
    return pd.read_csv(path, delimiter=delimiter)


def load_gas_price(path):
    return pd.read_csv(path)


def load_coal_price(path, delimiter=";"):
    return pd.read_csv(path, delimiter=delimiter)


def build_hourly_gas_price(gas_price_df, column="Gas (NCG, THE)"):
    """Expand a daily gas price series to hourly resolution (each day's price repeated 24x).

    The first row of the raw export is a stray header remnant and is dropped,
    matching the original notebook's per-year handling.
    """
    gas_price_df = gas_price_df.drop(gas_price_df.index[0])
    hourly_prices = [price for price in gas_price_df[column] for _ in range(24)]
    return pd.DataFrame({"gas_price": hourly_prices})


def build_hourly_coal_price(coal_df, start_year):
    """Expand a bi-weekly coal price series to hourly resolution.

    Averages every two consecutive quotes into one "monthly" figure, then
    broadcasts that figure across every hour of the corresponding month.
    """
    coal_df = coal_df.copy()
    coal_df["2_step"] = coal_df.index // 2

    # Clean the 'price' column: replace comma with dot and convert to numeric
    coal_df["price"] = coal_df["price"].astype(str).str.replace(",", ".", regex=False)
    coal_df["price"] = pd.to_numeric(coal_df["price"], errors="coerce")

    coal_avg = coal_df.groupby("2_step")["price"].mean().reset_index(drop=True)

    hourly_prices = []
    for index, monthly_avg in coal_avg.items():
        month = (index % 12) + 1
        year = start_year + (index // 12)
        num_days = pd.Period(f"{year}-{month}").days_in_month
        num_hours = num_days * 24
        hourly_prices.extend([monthly_avg] * num_hours)

    return pd.DataFrame({"hourly_coal_price": hourly_prices})
