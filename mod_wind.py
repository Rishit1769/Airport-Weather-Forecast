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

    if "datetime" in wind_df.columns:
        datetime_values = pd.Series(
            pd.to_datetime(wind_df["datetime"]),
            index=wind_df.index,
        )
    else:
        datetime_values = pd.Series(
            pd.to_datetime(wind_df.index),
            index=wind_df.index,
        )
    wind_df["time_gap_hrs"] = (
        datetime_values.diff().dt.total_seconds() / 3600.0
    )

    radians = np.radians(wind_df[direction_col])
    wind_df["u_wind"] = -wind_df["wind_speed"] * np.sin(radians)
    wind_df["v_wind"] = -wind_df["wind_speed"] * np.cos(radians)
    wind_df["ke"] = 0.5 * np.square(wind_df["wind_speed"])
    wind_df["ke_lag_1"] = wind_df["ke"].shift(1)

    # Causal spectral bands use only kinetic energy observed before this row.
    wind_df["ke_volatility_1h"] = wind_df["ke_lag_1"].rolling(2).std()
    wind_df["ke_volatility_6h"] = wind_df["ke_lag_1"].rolling(12).std()
    wind_df["ke_volatility_24h"] = wind_df["ke_lag_1"].rolling(48).std()
    wind_df["ke_divergence"] = (
        wind_df["ke_lag_1"].rolling(2).mean()
        - wind_df["ke_lag_1"].rolling(48).mean()
    )

    wind_df["u_lag_1"] = wind_df["u_wind"].shift(1)
    wind_df["v_lag_1"] = wind_df["v_wind"].shift(1)
    wind_df["speed_lag_1"] = wind_df["wind_speed"].shift(1)
    wind_df["u_shear_3h"] = wind_df["u_lag_1"] - wind_df["u_wind"].shift(6)
    wind_df["v_shear_3h"] = wind_df["v_lag_1"] - wind_df["v_wind"].shift(6)
    wind_df["total_shear_3h"] = np.sqrt(
        np.square(wind_df["u_shear_3h"])
        + np.square(wind_df["v_shear_3h"])
    )

    gap_mask_1 = wind_df["time_gap_hrs"] > 1.0
    wind_df.loc[
        gap_mask_1,
        ["u_lag_1", "v_lag_1", "speed_lag_1", "ke_lag_1"],
    ] = np.nan
    gap_mask_2 = wind_df["time_gap_hrs"].rolling(2).max() > 1.0
    gap_mask_6h = wind_df["time_gap_hrs"].rolling(12).max() > 1.0
    gap_mask_24h = wind_df["time_gap_hrs"].rolling(48).max() > 1.0
    wind_df.loc[
        gap_mask_2,
        ["ke_volatility_1h"],
    ] = np.nan
    wind_df.loc[
        gap_mask_6h,
        [
            "ke_volatility_6h",
            "u_shear_3h",
            "v_shear_3h",
            "total_shear_3h",
        ],
    ] = np.nan
    wind_df.loc[
        gap_mask_24h,
        ["ke_volatility_24h", "ke_divergence"],
    ] = np.nan

    if "pressure" in wind_df.columns:
        wind_df["pressure_diff_1h"] = wind_df["pressure"].diff(2)
        wind_df["pressure_diff_3h"] = wind_df["pressure"].diff(6)
        wind_df["max_gap_3h_window"] = wind_df["time_gap_hrs"].rolling(6).max()
        gap_mask_3h = wind_df["max_gap_3h_window"] > 1.0
        wind_df.loc[
            gap_mask_3h,
            ["pressure_diff_1h", "pressure_diff_3h"],
        ] = np.nan
        wind_df["pressure_volatility"] = wind_df["pressure_diff_3h"].abs()

    if "temp" in wind_df.columns:
        wind_df["temp_diff_1h"] = wind_df["temp"].diff(2)
        temp_gap_mask = wind_df["time_gap_hrs"].rolling(2).max() > 1.0
        wind_df.loc[temp_gap_mask, "temp_diff_1h"] = np.nan
        if "pressure_diff_1h" in wind_df.columns:
            wind_df["temp_roc_1h"] = wind_df["temp"].diff(2)
            wind_df["abl_instability"] = (
                wind_df["temp_roc_1h"]
                / (wind_df["pressure_diff_1h"] + 1e-5)
            )
            wind_df.loc[
                gap_mask_6h,
                ["temp_roc_1h", "abl_instability"],
            ] = np.nan

    hour_float = datetime_values.dt.hour + (datetime_values.dt.minute / 60.0)
    wind_df["sea_breeze_phase"] = np.sin(
        2.0 * np.pi * (hour_float - 14.0) / 24.0
    )

    required_history = ["u_lag_1", "ke_lag_1", "ke_volatility_24h"]
    if "pressure_diff_3h" in wind_df.columns:
        required_history.append("pressure_diff_3h")
    wind_df = wind_df.dropna(subset=required_history)
    return wind_df.drop(
        columns=["time_gap_hrs", "max_gap_3h_window"],
        errors="ignore",
    )


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
    return any(
        token in name
        for token in ("wind_speed", "wind_gust", "wind_dir", "humidity_wind", "low_wind")
    )


def _apply_anemometer_mask(df: pd.DataFrame) -> pd.DataFrame:
    """Remove periods where the physical wind sensor was locked or dead."""
    rolling_std = df["wind_speed"].rolling(12).std()
    dead_sensor = rolling_std < 0.01
    recovery = dead_sensor.shift(1, fill_value=False) & ~dead_sensor
    mask = dead_sensor | recovery

    purged = int(mask.sum())
    percentage = (purged / len(df) * 100.0) if len(df) else 0.0
    print(
        f"      -> Purged {purged} dead-sensor rows "
        f"({percentage:.2f}% of data)"
    )
    return df.loc[~mask].copy()


def train_and_predict(df_master: pd.DataFrame):
    wind_df = _apply_anemometer_mask(df_master)
    wind_df = add_wind_features(wind_df)
    direction_col = "wind_direction" if "wind_direction" in wind_df.columns else "wind_dir"

    split_idx = int(len(wind_df) * 0.85)
    train_df = wind_df.iloc[:split_idx].copy()
    test_df = wind_df.iloc[split_idx:].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Wind chronological split generated an empty partition.")

    y_train_u = train_df["u_wind"]
    y_train_v = train_df["v_wind"]
    y_train_speed = train_df["wind_speed"]

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
            "ke",
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
        "n_estimators": 2500,
        "learning_rate": 0.015,
        "max_depth": 8,
        "gamma": 0.1,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
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

    quantile_models = {}
    for alpha in (0.10, 0.50, 0.90):
        print(f"      -> Fitting Wind-Speed Quantile q={alpha:.2f}...")
        quantile_params = xgb_params.copy()
        quantile_params["objective"] = "reg:quantileerror"
        quantile_params["quantile_alpha"] = alpha
        model = XGBRegressor(**quantile_params)
        model.fit(X_train, y_train_speed, verbose=False)
        quantile_models[alpha] = model

    pred_u = _predict(model_u, X_test)
    pred_v = _predict(model_v, X_test)
    pred_q10 = _predict(quantile_models[0.10], X_test)
    pred_q50 = _predict(quantile_models[0.50], X_test)
    pred_q90 = _predict(quantile_models[0.90], X_test)

    # Guard against occasional quantile crossing before operational evaluation.
    pred_lower = np.clip(np.minimum(pred_q10, pred_q90), 0.0, 80.0)
    pred_upper = np.clip(np.maximum(pred_q10, pred_q90), 0.0, 80.0)
    pred_speed = np.clip(pred_q50, pred_lower, pred_upper)
    pred_gust = np.maximum(pred_speed, pred_upper)
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
    picp = float(
        np.mean((actual_speed >= pred_lower) & (actual_speed <= pred_upper))
    )
    mean_interval_width = float(np.mean(pred_upper - pred_lower))

    circular_error = np.abs(
        (pred_direction - actual_direction + 180.0) % 360.0 - 180.0
    )
    circular_mae = float(np.mean(circular_error))
    print(
        f"      -> Speed R2: {speed_metrics['r2']:.4f} | "
        f"Gust R2: {gust_metrics['r2']:.4f} | "
        f"Dir MAE: {circular_mae:.2f} deg | "
        f"PICP(10-90): {picp:.4f}"
    )

    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    joblib.dump(model_u, "checkpoints/wind_u_model.joblib")
    joblib.dump(model_v, "checkpoints/wind_v_model.joblib")
    for alpha, model in quantile_models.items():
        suffix = int(round(alpha * 100))
        joblib.dump(
            model,
            f"checkpoints/wind_speed_q{suffix:02d}_model.joblib",
        )

    return {
        "index": test_df.index,
        "wind_speed": {
            "y_true": actual_speed,
            "y_pred": pred_speed,
            "metrics": speed_metrics,
            "quantiles": {
                "q_0.10": pred_lower,
                "q_0.50": pred_speed,
                "q_0.90": pred_upper,
            },
            "picp_10_90": picp,
            "mean_interval_width": mean_interval_width,
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
