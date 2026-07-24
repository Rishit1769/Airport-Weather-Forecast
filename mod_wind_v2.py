from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from mod_wind import _apply_anemometer_mask, _metrics, _predict, add_wind_features
from nwp_fetch import fetch_nwp_history

VABB_LAT = 19.0887
VABB_LON = 72.8679
OVERLAP_START = "2017-01-01"
NWP_CACHE_PATH = "data/nwp_cache"


def _is_leaky_feature(column: str) -> bool:
    name = column.lower()
    if name.startswith("nwp_"):
        return False
    return any(token in name for token in ("wind_speed", "wind_gust", "wind_dir", "humidity_wind", "low_wind"))


def _add_nwp_features(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Wind MOS module requires a DatetimeIndex.")

    start_date = max(df.index.min().strftime("%Y-%m-%d"), OVERLAP_START)
    end_date = df.index.max().strftime("%Y-%m-%d")
    nwp_df = fetch_nwp_history(
        lat=VABB_LAT,
        lon=VABB_LON,
        start_date=start_date,
        end_date=end_date,
        cache_path=NWP_CACHE_PATH,
    )
    nwp_df = nwp_df.tz_convert(None)

    merged = df.join(nwp_df, how="left")
    merged = merged.loc[merged.index >= OVERLAP_START].copy()
    merged["nwp_pressure_diff"] = merged["pressure"] - merged["nwp_pressure"]
    merged["speed_residual"] = merged["wind_speed"] - merged["nwp_wind_speed"]
    merged["gust_residual"] = merged["wind_gust"] - merged["nwp_wind_gust"]
    merged = merged.dropna(
        subset=[
            "nwp_wind_speed",
            "nwp_wind_dir",
            "nwp_wind_dir_sin",
            "nwp_wind_dir_cos",
            "nwp_wind_gust",
            "nwp_pressure",
            "nwp_pressure_diff",
            "speed_residual",
            "gust_residual",
        ]
    )
    return merged


def _fit_quantile_model(X_train: pd.DataFrame, y_train: pd.Series, alpha: float) -> XGBRegressor:
    params = {
        "n_estimators": 2500,
        "learning_rate": 0.015,
        "max_depth": 8,
        "gamma": 0.1,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "tree_method": "hist",
        "device": "cuda",
        "objective": "reg:quantileerror",
        "quantile_alpha": alpha,
        "random_state": 42,
        "n_jobs": -1,
    }
    model = XGBRegressor(**params)
    model.fit(X_train, y_train, verbose=False)
    return model


def train_and_predict(df_master: pd.DataFrame):
    wind_df = _apply_anemometer_mask(df_master)
    wind_df = add_wind_features(wind_df)
    wind_df = _add_nwp_features(wind_df)
    direction_col = "wind_direction" if "wind_direction" in wind_df.columns else "wind_dir"

    split_idx = int(len(wind_df) * 0.85)
    train_df = wind_df.iloc[:split_idx].copy()
    test_df = wind_df.iloc[split_idx:].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Wind MOS chronological split generated an empty partition.")

    y_train_u = train_df["u_wind"]
    y_train_v = train_df["v_wind"]
    y_train_speed_residual = train_df["speed_residual"]
    y_train_gust_residual = train_df["gust_residual"]

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
            "speed_residual",
            "gust_residual",
        }
    ]
    features = list(
        dict.fromkeys(
            column
            for column in wind_df.columns
            if column not in drop_cols
            and not _is_leaky_feature(column)
            and pd.api.types.is_numeric_dtype(wind_df[column])
        )
    )

    X_train = train_df[features]
    X_test = test_df[features]

    shared_params = {
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

    print("      -> Fitting MOS Vector U Specialist...")
    model_u = XGBRegressor(**shared_params)
    model_u.fit(X_train, y_train_u, verbose=False)

    print("      -> Fitting MOS Vector V Specialist...")
    model_v = XGBRegressor(**shared_params)
    model_v.fit(X_train, y_train_v, verbose=False)

    quantile_models = {}
    for alpha in (0.10, 0.50, 0.90):
        print(f"      -> Fitting Residual Wind-Speed Quantile q={alpha:.2f}...")
        quantile_models[alpha] = _fit_quantile_model(X_train, y_train_speed_residual, alpha)

    print("      -> Fitting Gust Residual Specialist...")
    gust_model = XGBRegressor(**shared_params)
    gust_model.fit(X_train, y_train_gust_residual, verbose=False)

    pred_u = _predict(model_u, X_test)
    pred_v = _predict(model_v, X_test)
    pred_res_q10 = _predict(quantile_models[0.10], X_test)
    pred_res_q50 = _predict(quantile_models[0.50], X_test)
    pred_res_q90 = _predict(quantile_models[0.90], X_test)
    pred_gust_residual = _predict(gust_model, X_test)

    base_speed = test_df["nwp_wind_speed"].to_numpy(dtype=np.float64)
    base_gust = test_df["nwp_wind_gust"].to_numpy(dtype=np.float64)
    pred_q10 = np.clip(base_speed + pred_res_q10, 0.0, 80.0)
    pred_q50 = np.clip(base_speed + pred_res_q50, 0.0, 80.0)
    pred_q90 = np.clip(base_speed + pred_res_q90, 0.0, 80.0)
    pred_lower = np.minimum(pred_q10, pred_q90)
    pred_upper = np.maximum(pred_q10, pred_q90)
    pred_speed = np.clip(pred_q50, pred_lower, pred_upper)
    pred_gust = np.clip(base_gust + pred_gust_residual, 0.0, 80.0)
    pred_gust = np.maximum(pred_gust, pred_speed)
    pred_direction = (np.degrees(np.arctan2(-pred_u, -pred_v)) + 360.0) % 360.0

    actual_speed = test_df["wind_speed"].to_numpy(dtype=np.float64)
    actual_gust = test_df["wind_gust"].to_numpy(dtype=np.float64)
    actual_direction = test_df[direction_col].to_numpy(dtype=np.float64)
    actual_u = test_df["u_wind"].to_numpy(dtype=np.float64)
    actual_v = test_df["v_wind"].to_numpy(dtype=np.float64)

    speed_metrics = _metrics(actual_speed, pred_speed)
    gust_metrics = _metrics(actual_gust, pred_gust)
    nwp_speed_metrics = _metrics(actual_speed, base_speed)
    nwp_gust_metrics = _metrics(actual_gust, base_gust)
    u_metrics = _metrics(actual_u, pred_u)
    v_metrics = _metrics(actual_v, pred_v)
    picp = float(np.mean((actual_speed >= pred_lower) & (actual_speed <= pred_upper)))
    mean_interval_width = float(np.mean(pred_upper - pred_lower))

    circular_error = np.abs((pred_direction - actual_direction + 180.0) % 360.0 - 180.0)
    circular_mae = float(np.mean(circular_error))
    print(
        f"      -> MOS Speed R2: {speed_metrics['r2']:.4f} | "
        f"MOS Gust R2: {gust_metrics['r2']:.4f} | "
        f"Dir MAE: {circular_mae:.2f} deg | "
        f"NWP Speed R2: {nwp_speed_metrics['r2']:.4f}"
    )

    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    joblib.dump(model_u, "checkpoints/wind_v2_u_model.joblib")
    joblib.dump(model_v, "checkpoints/wind_v2_v_model.joblib")
    joblib.dump(gust_model, "checkpoints/wind_v2_gust_residual_model.joblib")
    for alpha, model in quantile_models.items():
        suffix = int(round(alpha * 100))
        joblib.dump(model, f"checkpoints/wind_v2_speed_residual_q{suffix:02d}_model.joblib")

    return {
        "index": test_df.index,
        "overlap_start": OVERLAP_START,
        "gust_target_kind": "synthetic_proxy_from_station_speed",
        "wind_speed": {
            "y_true": actual_speed,
            "y_pred": pred_speed,
            "baseline_pred": base_speed,
            "metrics": speed_metrics,
            "baseline_metrics": nwp_speed_metrics,
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
            "baseline_pred": base_gust,
            "metrics": gust_metrics,
            "baseline_metrics": nwp_gust_metrics,
        },
        "wind_dir": {
            "y_true": actual_direction,
            "y_pred": pred_direction,
            "circular_mae_deg": circular_mae,
            "component_metrics": {"u": u_metrics, "v": v_metrics},
        },
    }
