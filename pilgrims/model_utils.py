import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_metrics(y_true, y_pred):
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[valid]
    predicted = predicted[valid]

    absolute_error = np.abs(actual - predicted)
    denominator = np.abs(actual)
    nonzero = denominator > 1e-8
    smape_denominator = denominator + np.abs(predicted)
    smape_valid = smape_denominator > 1e-8

    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mape": float(np.mean(absolute_error[nonzero] / denominator[nonzero]) * 100)
        if nonzero.any() else None,
        "smape": float(np.mean(2 * absolute_error[smape_valid] / smape_denominator[smape_valid]) * 100)
        if smape_valid.any() else None,
        "wape": float(absolute_error.sum() / denominator.sum() * 100)
        if denominator.sum() > 1e-8 else None,
    }


def update_model_result(path, model_name, metrics, params=None, extra=None):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as file:
            results = json.load(file)
    else:
        results = []

    result = {"modelo": model_name, **metrics, "params": params or {}}
    if extra:
        result.update(extra)

    results = [item for item in results if item.get("modelo") != model_name]
    results.append(result)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=4)

    return result


def seasonal_naive(y_train, horizon, period):
    values = np.asarray(y_train, dtype=float)
    if len(values) < period:
        return np.repeat(values[-1], horizon)
    return np.resize(values[-period:], horizon)


def evaluate_baselines(y_train, y_test, seasonal_period):
    actual = np.asarray(y_test, dtype=float)
    last_value = np.repeat(float(np.asarray(y_train)[-1]), len(actual))
    seasonal_value = seasonal_naive(y_train, len(actual), seasonal_period)
    return {
        "naive": calculate_metrics(actual, last_value),
        "seasonal_naive": calculate_metrics(actual, seasonal_value),
    }


def walk_forward_predict(model_factory, X_train, y_train, X_test, y_test, step=50):
    """Generate expanding-window predictions with periodic model refits."""
    history_X = X_train.copy()
    history_y = y_train.copy()
    predictions = []

    for start in range(0, len(X_test), step):
        end = min(start + step, len(X_test))
        model = model_factory()
        model.fit(history_X, history_y)
        predictions.extend(model.predict(X_test.iloc[start:end]))
        history_X = pd.concat([history_X, X_test.iloc[start:end]])
        history_y = pd.concat([history_y, y_test.iloc[start:end]])

    return np.asarray(predictions)
