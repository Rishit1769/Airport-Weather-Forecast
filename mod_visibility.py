import logging

import joblib
import numpy as np
import pandas as pd

from model_common import prepare_target, predict_regressor, regression_metrics, train_regressor

logger = logging.getLogger(__name__)


def train_and_predict(df: pd.DataFrame):
    train, val, test, features = prepare_target(
        df,
        "visibility",
        transform=lambda values: np.sqrt(values.clip(150.0, 10000.0)),
    )
    model = train_regressor(
        train[features],
        train["visibility_target"],
        val[features],
        val["visibility_target"],
    )
    y_true = np.square(test["visibility_target"].to_numpy(dtype=np.float64))
    y_pred = np.clip(np.square(predict_regressor(model, test[features])), 150.0, 10000.0)
    joblib.dump(model, "checkpoints/visibility_target_model.joblib")
    metrics = regression_metrics(y_true, y_pred)
    logger.info("Visibility metrics: %s", metrics)
    return {"y_true": y_true, "y_pred": y_pred, "index": test.index, "metrics": metrics}

