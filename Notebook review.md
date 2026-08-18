---
name: notebook-review
description: Review of Battery_optimization_forecasting.ipynb (github.com/Nick300500/Electricity-Price-Forecast-Battery-Optimization) — architecture, data flow, and identified issues to fix before building a rolling/live version.
---

# Notebook: Battery_optimization_forecasting.ipynb

Repo: https://github.com/Nick300500/Electricity-Price-Forecast-Battery-Optimization
Reviewed: 2026-08-18 (74 cells, pulled via raw.githubusercontent.com)

## What it does (pipeline)

1. **Storage optimizer** (`optimize_storage_dispatch`, pyomo/HiGHS): given a price series, solves for optimal charge/discharge schedule of a battery (default params modeled roughly on a Sungrow SBR096, 3 modules: capacity 9.6 kWh, p_max 5.76 kW, eff_ch/dis 0.95). Verified qualitatively — charges at local price minima, discharges at maxima.
2. **Feature dataset**: merges Day-Ahead price (2023–2025 H1), CO2 auction prices (DE/EU), gas price (NCG/THE, forward-filled to hourly), coal price (monthly avg, forward-filled to hourly), grid load, and generation by source (wind onshore/offshore, solar) — plus calendar features (weekday/hour/month etc.).
3. **Model**: RandomForestRegressor (n_estimators=180, max_depth=10, tuned via GridSearchCV). Trained on 2023–2024 data, evaluated on H1 2025 held out as "testrun_df".
4. **Backtest chain**: perfect-foresight profit (actual prices → optimizer) vs. forecast-driven profit (predicted prices → optimizer schedule → applied to actual prices for real P&L) vs. sensitivity analysis (forecast artificially shifted ±1 €/MWh and ±10% toward/away from actual, plus a messy ±10–90% sweep at the end).
5. Simple payback-period calc assuming ~2930 € system cost.

## Issues found (before extending toward rolling/live operation)

1. **Likely data-leakage bug in the RF fit (cell 20).** After the train/val split, the code calls `rf_model.fit(features, target)` — the *full* dataset, not `X_train, y_train`. So the model has already seen the validation rows during training, and `mae_val_rf` etc. are not a clean out-of-sample estimate. This needs fixing before trusting any reported MAE/RMSE.
2. **GridSearchCV likely leaks across time.** `GridSearchCV(..., cv=3)` uses default KFold, which shuffles/splits without respecting chronological order. For a time-indexed target this lets future data help predict the past inside each fold. Should use `TimeSeriesSplit` (or a purged/blocked CV) instead.
3. **Feature realism for live deployment: several inputs are actuals, not day-ahead forecasts.** `wind_gen_off/on`, `solar_gen`, and `net_load` are pulled from "Berechnete Auflösungen" (calculated/actual generation) columns, and gas/coal prices are monthly settlement prices. In a live system you only have day-ahead *forecasts* of generation/load at bidding time, not the realized values — using actuals as features overstates achievable accuracy and won't be reproducible operationally. This is probably the single most important fix if the goal is to actually run a battery on it.
4. **Notebook isn't a reproducible top-to-bottom pipeline yet.** Many cells depend on execution order and on files written to `Results/` by earlier cells (read-your-own-write pattern), some CSV column/index assumptions look brittle (hardcoded `skiprows`, mixed date formats, `index_col=4` in one place). Also cell 8 re-loads `price_data.csv` separately/redundantly from the cell-13 loading path. Fine for exploratory notebook, not fine for something that should run unattended daily.
5. **No rolling/walk-forward structure yet** — single train/test split on a block of history. That's the explicit next step the user wants ("rolling aufbauen").
6. **Coal price is monthly, broadcast flat across every hour of the month** — reasonable given data availability, but means it contributes near-zero intra-day signal; worth checking its actual feature importance vs. complexity cost.

## What the user wants next (stated intent)

- Make the model more robust ("gebauer machen") — possibly rolling-window retraining.
- Eventually use it to actually operate a home battery (not just backtest).
- Working with Claude Code on the implementation.
