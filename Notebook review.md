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

1. ~~**Likely data-leakage bug in the RF fit (cell 20).**~~ **Fixed 2026-08-18.** `modeling.train_random_forest` now fits on `X_train`/`y_train` only; `mae_val`/`rmse_val` are a clean out-of-sample estimate (validation MAE moved from 7.67 to ~22-23 €/MWh once the leak was closed — the old number was not trustworthy).
2. ~~**GridSearchCV likely leaks across time.**~~ **Fixed 2026-08-18.** `modeling.tune_random_forest` now uses `TimeSeriesSplit` instead of plain K-fold.
3. ~~**Feature realism for live deployment: several inputs are actuals, not day-ahead forecasts.**~~ **Partially fixed 2026-08-18.** `wind_gen_off/on`, `solar_gen`, `other_gen`, `total_gen` now come from SMARD's day-ahead generation *forecasts* (`battery_opt/smard_client.py`, filter ids confirmed against `bundesAPI/smard-api` + live probing — note the community spec has a typo, PV is filter `125` not `126` as its own enum implies). `net_load`/`res_load` are **still on realized values** — no confirmed SMARD filter id for a day-ahead load forecast was found; see the comment on `smard_client.FORECAST_LOAD_FILTER` for how to fill it in once one's located. Gas/coal prices are still monthly settlement prices (unchanged, reasonable given availability).
4. **Notebook isn't a reproducible top-to-bottom pipeline yet.** Partially addressed: `scripts/run_pipeline.py` / `battery_opt/pipeline.py` now run the whole thing end-to-end from the terminal without any read-your-own-write CSV round-tripping, and is the intended way to actually run this going forward. The notebook itself still has the original brittleness (hardcoded `skiprows`, mixed date formats, `index_col=4`) since it's being kept as an exploratory/analysis surface, not the operational path.
5. **No rolling/walk-forward structure yet** — single train/test split on a block of history. That's the explicit next step the user wants ("rolling aufbauen").
6. **Coal price is monthly, broadcast flat across every hour of the month** — reasonable given data availability, but means it contributes near-zero intra-day signal; worth checking its actual feature importance vs. complexity cost.

## What the user wants next (stated intent)

- Make the model more robust ("gebauer machen") — possibly rolling-window retraining.
- Eventually use it to actually operate a home battery (not just backtest).
- Working with Claude Code on the implementation.
