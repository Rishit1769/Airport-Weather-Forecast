import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

SEED = 42
HORIZON = 12
TRAIN_START = "2016-01-01"
TRAIN_END = "2024-01-01"
VAL_END = "2025-01-01"
TEST_END = "2026-01-01"

_RUNTIME: Dict[str, str] | None = None


def get_runtime() -> Dict[str, str]:
    global _RUNTIME
    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    if _RUNTIME is None:
        _RUNTIME = {"tree_method": "hist", "device": "cuda"}
        probe = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=1,
            max_depth=1,
            learning_rate=0.3,
            **_RUNTIME,
        )
        probe.fit(pd.DataFrame({"x": [0.0, 1.0]}), pd.Series([0.0, 1.0]), verbose=False)
        logger.info("XGBoost runtime: %s", _RUNTIME)
    return _RUNTIME


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    selected = []
    for col in df.columns:
        if col.endswith("_target") or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        std_val = float(df[col].std(ddof=0))
        if np.isfinite(std_val) and std_val >= 1e-6 and float(df[col].isna().mean()) <= 0.30:
            selected.append(col)
    return selected


def prepare_target(
    df: pd.DataFrame,
    source_col: str,
    transform=None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    work = df.copy()
    target = work[source_col].shift(-HORIZON)
    if transform is not None:
        target = transform(target)
    work[f"{source_col}_target"] = target
    work = work.dropna().copy()

    train = work.loc[(work.index >= TRAIN_START) & (work.index < TRAIN_END)].copy()
    val = work.loc[(work.index >= TRAIN_END) & (work.index < VAL_END)].copy()
    test = work.loc[(work.index >= VAL_END) & (work.index < TEST_END)].copy()
    if train.empty or val.empty or test.empty:
        raise ValueError(f"Chronological split for {source_col} generated an empty partition.")
    return train, val, test, get_feature_columns(work)


def train_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
):
    runtime = get_runtime()
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1500,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=50,
        random_state=SEED,
        n_jobs=-1,
        tree_method=runtime["tree_method"],
        device=runtime["device"],
        gamma=0.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def predict_regressor(model, X: pd.DataFrame) -> np.ndarray:
    matrix = xgb.DMatrix(X, enable_categorical=True)
    best_iteration = getattr(model, "best_iteration", None)
    iteration_range = (0, int(best_iteration) + 1) if best_iteration is not None else (0, 0)
    return np.asarray(
        model.get_booster().predict(matrix, iteration_range=iteration_range, strict_shape=False),
        dtype=np.float64,
    )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
