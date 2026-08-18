# Electricity-Price-Forecast-Battery-Optimization

Forecasts German Day-Ahead electricity prices and backtests a residential battery's charge/discharge profit against that forecast.

## Structure

- `Battery_optimization_forecasting.ipynb` — orchestrates the pipeline (load → prepare → train → backtest → sensitivity analysis). Each stage is a thin call into `src/battery_opt`.
- `src/battery_opt/` — the actual pipeline logic:
  - `optimizer.py` — pyomo/HiGHS battery dispatch optimizer
  - `data_loading.py` — per-source CSV loaders (price, generation, load, CO2, gas, coal)
  - `data_prep.py` — merges raw sources into the feature dataset
  - `modeling.py` — train/tune/evaluate/persist the RandomForest price forecast
  - `backtest.py` — perfect-foresight vs. forecast-driven profit, sensitivity-analysis price corrections, payback calculation
  - `plotting.py` — shared plot helpers
- `Notebook review.md` — review notes / known issues to address next (data leakage, CV strategy, forecast realism for live use, etc.)

## Setup

```
pip install -r requirements.txt
```

The notebook expects the source CSVs (price, generation, load, CO2, gas, coal for 2023-2025) in the repo root.
