# Electricity-Price-Forecast-Battery-Optimization

Forecasts German Day-Ahead electricity prices and backtests a residential battery's charge/discharge profit against that forecast.

## Structure

- `scripts/run_pipeline.py` — terminal entry point: load → prepare → train → forecast → dispatch → signal report. This is the way to actually run the pipeline right now.
- `Battery_optimization_forecasting.ipynb` — exploratory/analysis notebook (plots, sensitivity analysis, payback calc). Same underlying package, but not the primary way to run things anymore.
- `src/battery_opt/` — the actual pipeline logic:
  - `optimizer.py` — pyomo/HiGHS battery dispatch optimizer
  - `data_loading.py` — per-source CSV loaders (price, generation, load, CO2, gas, coal)
  - `data_prep.py` — merges raw sources into the feature dataset
  - `modeling.py` — train/tune/evaluate/persist the RandomForest price forecast
  - `backtest.py` — perfect-foresight vs. forecast-driven profit, sensitivity-analysis price corrections, payback calculation, `build_signal_report()` for the battery signal CSV
  - `plotting.py` — shared plot helpers (used by the notebook)
  - `pipeline.py` — orchestrates the stages above end to end; used by `scripts/run_pipeline.py` and meant to become the basis for a persistent/rolling-retrain service later
  - `smard_client.py` — minimal client for SMARD.de's public chart_data API; fetches day-ahead generation forecasts (see Known issues)
- `scripts/fetch_forecast_data.py` — pre-fetches SMARD generation-forecast CSVs into `Data/`. `pipeline.py` also calls this automatically, so this is only needed to warm the cache on its own.
- `Data/` — raw source CSVs (price, generation, load, CO2, gas, coal for 2023-2025), plus `gen_forecast_*.csv` (auto-fetched). Not in version control — see below.
- `Notebook review.md` — review notes / known issues to address next (walk-forward retraining, load-forecast gap, etc.)

## Setup

A local venv named "Battery Optimization" (registered as the Jupyter kernel `battery-optimization`) already has everything installed:

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Put the raw source CSVs in `Data/` (not tracked in git — ask about where to source them if you don't have them).

## Running

```
.venv\Scripts\python.exe scripts\run_pipeline.py
```

Loads the data, trains the forecast model, dispatches the battery off the 2025 forecast, and writes `Results/battery_signal_report_2025.csv` (time, P_ch, P_dis, E, predicted_price, actual_price, profit). Takes about 15-20 seconds. `Results/`, `Extra/`, `outputs/` (the pickled model) are all gitignored — pipeline output, not source.

## Known issues

See `Notebook review.md`. Fixed so far: RF data leakage (now trained on the train split only), `GridSearchCV` now uses `TimeSeriesSplit`, and wind/solar/other generation features now come from SMARD's day-ahead forecasts instead of realized values. Still open: `net_load`/`res_load` stay on realized values (no confirmed SMARD day-ahead load-forecast filter id — see the comment on `smard_client.FORECAST_LOAD_FILTER`), and there's no rolling/walk-forward retraining yet.
