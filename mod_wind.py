import logging

import joblib
import numpy as np
import pandas as pd

from model_common import prepare_target, predict_regressor, regression_metrics, train_regressor

logger = logging.getLogger(__name__)


def _train_component(df: pd.DataFrame, source_col: str, bounds: tuple[float, float]):
    train, val, test, features = prepare_target(df, source_col)
    target_col = f"{source_col}_target"
    model = train_regressor(train[features], train[target_col], val[features], val[target_col])
    y_true = test[target_col].to_numpy(dtype=np.float64)
    y_pred = np.clip(predict_regressor(model, test[features]), *bounds)
    joblib.dump(model, f"checkpoints/{target_col}_model.joblib")
    return y_true, y_pred, test.index, regression_metrics(y_true, y_pred)


def train_and_predict(df: pd.DataFrame):
    speed_true, speed_pred, index, speed_metrics = _train_component(df, "wind_speed", (0.0, 80.0))
    gust_true, gust_pred, _, gust_metrics = _train_component(df, "wind_gust", (0.0, 80.0))
    sin_true, sin_pred, _, sin_metrics = _train_component(df, "wind_dir_sin", (-1.0, 1.0))
    cos_true, cos_pred, _, cos_metrics = _train_component(df, "wind_dir_cos", (-1.0, 1.0))

    direction_true = np.degrees(np.arctan2(sin_true, cos_true)) % 360.0
    direction_pred = np.degrees(np.arctan2(sin_pred, cos_pred)) % 360.0
    circular_error = np.abs((direction_pred - direction_true + 180.0) % 360.0 - 180.0)
    circular_mae = float(np.mean(circular_error))
    logger.info("Wind direction mean circular error: %.3f degrees", circular_mae)

    return {
        "index": index,
        "wind_speed": {
            "y_true": speed_true,
            "y_pred": speed_pred,
            "metrics": speed_metrics,
        },
        "wind_gust": {
            "y_true": gust_true,
            "y_pred": gust_pred,
            "metrics": gust_metrics,
        },
        "wind_dir": {
            "y_true": direction_true,
            "y_pred": direction_pred,
            "circular_mae_deg": circular_mae,
            "component_metrics": {"sin": sin_metrics, "cos": cos_metrics},
        },
    }

