"""Battery storage dispatch optimizer (pyomo/HiGHS).

Given a price time series, solves for the profit-maximizing charge/discharge
schedule of a battery with the given storage parameters.
"""

import logging

import pyomo.environ as pyo
import pandas as pd

from .config import DEFAULT_STORAGE_PARAMS

# HiGHS logs its full solve trace (LP size, simplex iterations, ...) through
# this logger at INFO level, which drowns out the pipeline's own progress
# logging on every single dispatch call. It's a genuine solver diagnostic,
# not noise from our code, so only raise the bar rather than disabling it.
logging.getLogger("pyomo.contrib.appsi.solvers.highs").setLevel(logging.WARNING)


def optimize_storage_dispatch(price_df, storage_params=None):
    """
    Optimizes battery storage dispatch to maximize profit given external price time series.

    Returns
    -------
    model : pyomo.ConcreteModel
        Solved Pyomo model.
    """
    if storage_params is None:
        storage_params = DEFAULT_STORAGE_PARAMS

    # Ensure price_df is a Series
    if isinstance(price_df, pd.DataFrame):
        price_series = price_df.iloc[:, 0]
    else:
        price_series = price_df

    n_steps = len(price_series)
    model = pyo.ConcreteModel()
    model.T = pyo.RangeSet(0, n_steps - 1)

    # Variables
    model.P_ch = pyo.Var(
        model.T, within=pyo.NonNegativeReals, bounds=(0, storage_params["p_max"])
    )
    model.P_dis = pyo.Var(
        model.T, within=pyo.NonNegativeReals, bounds=(0, storage_params["p_max"])
    )
    model.E = pyo.Var(
        model.T, within=pyo.NonNegativeReals, bounds=(0, storage_params["capacity"])
    )

    # Storage state of charge constraints
    def soc_rule(model, t):
        if t == 0:
            return (
                model.E[t]
                == storage_params["soc_init"]
                + model.P_ch[t] * storage_params["eff_ch"]
                - model.P_dis[t] / storage_params["eff_dis"]
            )
        else:
            return (
                model.E[t]
                == model.E[t - 1]
                + model.P_ch[t] * storage_params["eff_ch"]
                - model.P_dis[t] / storage_params["eff_dis"]
            )

    model.soc = pyo.Constraint(model.T, rule=soc_rule)

    # Objective: maximize profit (revenue from discharge - cost of charge)
    model.obj = pyo.Objective(
        expr=sum(
            price_series.iat[t] * model.P_dis[t] - price_series.iat[t] * model.P_ch[t]
            for t in model.T
        ),
        sense=pyo.maximize,
    )

    # Solve
    solver = pyo.SolverFactory("appsi_highs")
    solver.solve(model)

    return model


def extract_storage_profits(model, price_series, storage_params=None):
    """
    Processes the solved model and outputs a DataFrame with dispatch and profit per time step.
    """
    if storage_params is None:
        storage_params = DEFAULT_STORAGE_PARAMS

    n_steps = len(price_series)
    results_df = pd.DataFrame(
        {
            "P_ch": [pyo.value(model.P_ch[t]) for t in range(n_steps)],
            "P_dis": [pyo.value(model.P_dis[t]) for t in range(n_steps)],
            "E": [pyo.value(model.E[t]) for t in range(n_steps)],
            "price": [price_series.iat[t] for t in range(n_steps)],
        },
        index=price_series.index,
    )
    results_df["profit"] = (
        results_df["price"] * results_df["P_dis"] - results_df["price"] * results_df["P_ch"]
    )

    return results_df


def run_dispatch(price_series, storage_params=None):
    """Convenience wrapper: optimize and extract profits in one call."""
    model = optimize_storage_dispatch(price_series, storage_params)
    return extract_storage_profits(model, price_series, storage_params)
