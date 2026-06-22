import logging

import joblib
import numpy as np
import pandas as pd

from model_common import prepare_target, predict_regressor, regression_metrics, train_regressor

logger = logging.getLogger(__name__)


def train_and_predict(df: pd.DataFrame):
    train, val, test, features = prepare_target(df, "pressure")
    model = train_regressor(
        train[features],
        train["pressure_target"],
        val[features],
        val["pressure_target"],
    )
    y_true = test["pressure_target"].to_numpy(dtype=np.float64)
    y_pred = np.clip(predict_regressor(model, test[features]), 950.0, 1050.0)
    joblib.dump(model, "checkpoints/pressure_target_model.joblib")
    metrics = regression_metrics(y_true, y_pred)
    logger.info("Pressure metrics: %s", metrics)
    return {"y_true": y_true, "y_pred": y_pred, "index": test.index, "metrics": metrics}

