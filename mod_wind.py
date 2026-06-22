from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


def add_wind_features(df: pd.DataFrame) -> pd.DataFrame:
    wind_df = df.copy()
    direction_col = "wind_direction" if "wind_direction" in wind_df.columns else "wind_dir"

    radians = np.radians(wind_df[direction_col])
    wind_df["u_wind"] = -wind_df["wind_speed"] * np.sin(radians)
    wind_df["v_wind"] = -wind_df["wind_speed"] * np.cos(radians)

    wind_df["u_lag_1"] = wind_df["u_wind"].shift(1)
    wind_df["v_lag_1"] = wind_df["v_wind"].shift(1)
    wind_df["speed_lag_1"] = wind_df["wind_speed"].shift(1)

    if "pressure" in wind_df.columns:
        wind_df["pressure_diff_1h"] = wind_df["pressure"].diff(2)
        wind_df["pressure_diff_3h"] = wind_df["pressure"].diff(6)

    if "temp" in wind_df.columns:
        wind_df["temp_diff_1h"] = wind_df["temp"].diff(2)

    if "datetime" in wind_df.columns:
        datetime_values = pd.to_datetime(wind_df["datetime"])
    else:
        datetime_values = wind_df.index
    hour_float = datetime_values.hour + (datetime_values.minute / 60.0)
    wind_df["sea_breeze_phase"] = np.sin(
        2.0 * np.pi * (hour_float - 14.0) / 24.0
    )

    return wind_df.dropna(subset=["u_lag_1"])


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def _predict(model: XGBRegressor, features: pd.DataFrame) -> np.ndarray:
    matrix = xgb.DMatrix(features, enable_categorical=True)
    return np.asarray(
        model.get_booster().predict(matrix, strict_shape=False),
        dtype=np.float64,
    )


def _is_leaky_wind_feature(column: str) -> bool:
    """Reject shared features that encode the current wind observation."""
    name = column.lower()
    safe_wind_prefixes = (
        "wind_speed_lag_",
        "wind_gust_lag_",
        "wind_dir_sin_lag_",
        "wind_dir_cos_lag_",
        "wind_speed_change_lag_",
    )
    if name.startswith(safe_wind_prefixes):
        return False
    return any(
        token in name
        for token in ("wind_speed", "wind_gust", "wind_dir", "humidity_wind", "low_wind")
    )


def train_and_predict(df_master: pd.DataFrame):
    wind_df = add_wind_features(df_master)
    direction_col = "wind_direction" if "wind_direction" in wind_df.columns else "wind_dir"

    split_idx = int(len(wind_df) * 0.85)
    train_df = wind_df.iloc[:split_idx].copy()
    test_df = wind_df.iloc[split_idx:].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Wind chronological split generated an empty partition.")

    y_train_u = train_df["u_wind"]
    y_train_v = train_df["v_wind"]
    y_train_gust_delta = train_df["wind_gust"] - train_df["wind_speed"]

    drop_cols = [
        column
        for column in wind_df.columns
        if "target" in column.lower()
        or column
        in {
            "wind_speed",
            "wind_gust",
            "wind_direction",
            "wind_dir",
            "u_wind",
            "v_wind",
            "wind_dir_sin",
            "wind_dir_cos",
            "datetime",
        }
    ]
    features = list(
        dict.fromkeys(
            column
            for column in wind_df.columns
            if column not in drop_cols
            and not _is_leaky_wind_feature(column)
            and pd.api.types.is_numeric_dtype(wind_df[column])
        )
    )

    X_train = train_df[features]
    X_test = test_df[features]

    xgb_params = {
        "n_estimators": 1500,
        "learning_rate": 0.02,
        "max_depth": 7,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "device": "cuda",
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": -1,
    }

    print("      -> Fitting Vector U Specialist...")
    model_u = XGBRegressor(**xgb_params)
    model_u.fit(X_train, y_train_u, verbose=False)

    print("      -> Fitting Vector V Specialist...")
    model_v = XGBRegressor(**xgb_params)
    model_v.fit(X_train, y_train_v, verbose=False)

    print("      -> Fitting Gust Delta Specialist...")
    model_gust = XGBRegressor(**xgb_params)
    model_gust.fit(X_train, y_train_gust_delta, verbose=False)

    pred_u = _predict(model_u, X_test)
    pred_v = _predict(model_v, X_test)
    pred_gust_delta = np.clip(_predict(model_gust, X_test), 0.0, None)

    pred_speed = np.clip(np.sqrt(np.square(pred_u) + np.square(pred_v)), 0.0, 80.0)
    pred_gust = np.clip(pred_speed + pred_gust_delta, pred_speed, 80.0)
    pred_direction = (
        np.degrees(np.arctan2(-pred_u, -pred_v)) + 360.0
    ) % 360.0

    actual_speed = test_df["wind_speed"].to_numpy(dtype=np.float64)
    actual_gust = test_df["wind_gust"].to_numpy(dtype=np.float64)
    actual_direction = test_df[direction_col].to_numpy(dtype=np.float64)
    actual_u = test_df["u_wind"].to_numpy(dtype=np.float64)
    actual_v = test_df["v_wind"].to_numpy(dtype=np.float64)

    speed_metrics = _metrics(actual_speed, pred_speed)
    gust_metrics = _metrics(actual_gust, pred_gust)
    u_metrics = _metrics(actual_u, pred_u)
    v_metrics = _metrics(actual_v, pred_v)

    circular_error = np.abs(
        (pred_direction - actual_direction + 180.0) % 360.0 - 180.0
    )
    circular_mae = float(np.mean(circular_error))
    print(
        f"      -> Speed R2: {speed_metrics['r2']:.4f} | "
        f"Gust R2: {gust_metrics['r2']:.4f} | "
        f"Dir MAE: {circular_mae:.2f} deg"
    )

    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    joblib.dump(model_u, "checkpoints/wind_u_model.joblib")
    joblib.dump(model_v, "checkpoints/wind_v_model.joblib")
    joblib.dump(model_gust, "checkpoints/wind_gust_delta_model.joblib")

    return {
        "index": test_df.index,
        "wind_speed": {
            "y_true": actual_speed,
            "y_pred": pred_speed,
            "metrics": speed_metrics,
        },
        "wind_gust": {
            "y_true": actual_gust,
            "y_pred": pred_gust,
            "metrics": gust_metrics,
        },
        "wind_dir": {
            "y_true": actual_direction,
            "y_pred": pred_direction,
            "circular_mae_deg": circular_mae,
            "component_metrics": {"u": u_metrics, "v": v_metrics},
        },
    }
