import logging

import joblib
import numpy as np
import pandas as pd

from model_common import prepare_target, predict_regressor, regression_metrics, train_regressor

logger = logging.getLogger(__name__)


def train_and_predict(df: pd.DataFrame):
    train, val, test, features = prepare_target(df, "temp")
    model = train_regressor(
        train[features],
        train["temp_target"],
        val[features],
        val["temp_target"],
    )
    y_true = test["temp_target"].to_numpy(dtype=np.float64)
    y_pred = np.clip(predict_regressor(model, test[features]), -10.0, 55.0)
    joblib.dump(model, "checkpoints/temp_target_model.joblib")
    metrics = regression_metrics(y_true, y_pred)
    logger.info("Temperature metrics: %s", metrics)
    return {"y_true": y_true, "y_pred": y_pred, "index": test.index, "metrics": metrics}

