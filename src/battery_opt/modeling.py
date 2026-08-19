"""Random-forest Day-Ahead price forecast: split, train, tune, persist."""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split

from .config import RF_MODEL_PATH

RANDOM_STATE = 42


def prepare_features_target(forecasting_data):
    """Drop non-numeric columns and split into features / target ('price')."""
    df = forecasting_data.replace("-", np.nan)
    df = df.drop(columns=["time"]).astype(float)
    features = df.drop("price", axis=1)
    target = df["price"]
    return features, target


def train_val_split(features, target, test_size=0.1, random_state=RANDOM_STATE):
    # shuffle=False: chronological order is kept to avoid mixing future/past rows.
    return train_test_split(features, target, test_size=test_size, random_state=random_state, shuffle=False)


def train_random_forest(X_train, y_train, X_val, y_val, n_estimators=180, max_depth=10,
                         random_state=RANDOM_STATE, model_path=RF_MODEL_PATH):
    """Fit the RandomForestRegressor on the training split only and report
    train/validation MAE & RMSE (validation is a clean out-of-sample estimate)."""
    rf_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    rf_model.fit(X_train, y_train)

    y_pred_train = rf_model.predict(X_train)
    y_pred_val = rf_model.predict(X_val)

    metrics = {
        "mae_train": mean_absolute_error(y_train, y_pred_train),
        "rmse_train": np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "mae_val": mean_absolute_error(y_val, y_pred_val),
        "rmse_val": np.sqrt(mean_squared_error(y_val, y_pred_val)),
    }

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(rf_model, f)

    return rf_model, y_pred_train, y_pred_val, metrics


def load_random_forest(model_path=RF_MODEL_PATH):
    with open(model_path, "rb") as f:
        return pickle.load(f)


def tune_random_forest(X_train, y_train, X_val, y_val, n_estimators_range=range(100, 201, 10),
                        max_depth_range=range(5, 11, 1), n_splits=3, random_state=RANDOM_STATE):
    """Grid-search over n_estimators/max_depth using TimeSeriesSplit CV.

    Plain K-fold would shuffle/split without respecting chronological order,
    letting future rows help predict past ones within a fold. TimeSeriesSplit
    only ever validates on data that comes after its training fold.
    """
    param_grid = {
        "n_estimators": list(n_estimators_range),
        "max_depth": list(max_depth_range),
    }
    grid_rf = GridSearchCV(
        RandomForestRegressor(random_state=random_state),
        param_grid,
        cv=TimeSeriesSplit(n_splits=n_splits),
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    grid_rf.fit(X_train, y_train)

    best_rf = grid_rf.best_estimator_
    y_pred_val_best = best_rf.predict(X_val)
    tuned_mae = mean_absolute_error(y_val, y_pred_val_best)

    return grid_rf, best_rf, tuned_mae


def evaluate(actual, predicted):
    """Return (MAE, RMSE) for a set of predictions."""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return mae, rmse


def feature_importances(rf_model, feature_names):
    return pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=False)


def forecast_prices(rf_model, testrun_df, feature_columns):
    """Predict prices for testrun_df, returning a DataFrame indexed by 'time'."""
    X_forecast = testrun_df[feature_columns]
    predicted = rf_model.predict(X_forecast)
    return pd.DataFrame({"forecasted_price": predicted}, index=testrun_df["time"])
