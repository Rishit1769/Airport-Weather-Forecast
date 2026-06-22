from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


def add_vis_features(df: pd.DataFrame) -> pd.DataFrame:
    vis_df = df.copy()

    vis_df["vis_lag_1"] = vis_df["visibility"].shift(1)
    vis_df["temp_lag_1"] = vis_df["temp"].shift(1)
    vis_df["temp_lag_3"] = vis_df["temp"].shift(3)

    if "dew_point" in vis_df.columns:
        dew_point = vis_df["dew_point"].shift(1)
    elif "humidity" in vis_df.columns:
        dew_point = vis_df["temp_lag_1"] - ((100.0 - vis_df["humidity"].shift(1)) / 5.0)
    else:
        dew_point = vis_df["temp_lag_1"] - 5.0

    vis_df["dew_depression"] = vis_df["temp_lag_1"] - dew_point
    vis_df["dew_dep_squared"] = np.square(vis_df["dew_depression"])
    vis_df["temp_cooling_rate"] = vis_df["temp_lag_1"] - vis_df["temp_lag_3"]

    if "wind_speed" in vis_df.columns:
        vis_df["wind_lag_1"] = vis_df["wind_speed"].shift(1)
    if "wind_speed" in vis_df.columns and "pressure" in vis_df.columns:
        vis_df["stagnation_index"] = vis_df["pressure"].shift(1) / (
            vis_df["wind_speed"].shift(1) + 1.0
        )

    if "datetime" in vis_df.columns:
        datetime_values = pd.to_datetime(vis_df["datetime"])
    else:
        datetime_values = vis_df.index
    vis_df["is_fog_time"] = (
        (datetime_values.hour >= 2) & (datetime_values.hour <= 8)
    ).astype(int)

    return vis_df.dropna(subset=["vis_lag_1", "temp_lag_3"])


def apply_vis_outage_mask(df: pd.DataFrame) -> pd.DataFrame:
    """Flags and removes forward-filled sensor outages."""
    vis_diff = df["visibility"].diff().abs()
    is_flatline = vis_diff.rolling(window=10).sum().eq(0.0)
    return df.loc[~is_flatline].copy()


def train_and_predict(df_master: pd.DataFrame):
    vis_df = add_vis_features(df_master)
    rows_before_mask = len(vis_df)
    vis_df = apply_vis_outage_mask(vis_df)
    print(f"      -> Visibility outage rows removed: {rows_before_mask - len(vis_df)}")
    vis_df = vis_df.dropna().copy()

    split_idx = int(len(vis_df) * 0.85)
    train_df = vis_df.iloc[:split_idx].copy()
    test_df = vis_df.iloc[split_idx:].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Visibility chronological split generated an empty partition.")

    y_train_delta = np.sqrt(train_df["visibility"]) - np.sqrt(train_df["vis_lag_1"])
    y_test_abs = test_df["visibility"]

    unsafe_visibility_features = {
        "visibility_trend",
        "visibility_acceleration",
        "vis_drop_1",
        "vis_drop_3",
        "vis_drop_rate",
        "vis_regime",
        "low_visibility_flag",
        "low_visibility_streak",
    }
    drop_cols = [
        col
        for col in vis_df.columns
        if (
            "target" in col.lower()
            or col == "visibility"
            or col == "datetime"
            or col in unsafe_visibility_features
        )
    ]
    features = [
        col
        for col in vis_df.columns
        if col not in drop_cols
        and pd.api.types.is_numeric_dtype(vis_df[col])
        and (
            "_lag_" in col.lower()
            or col
            in {
                "hour_sin",
                "hour_cos",
                "month_sin",
                "month_cos",
                "dew_depression",
                "dew_dep_squared",
                "temp_cooling_rate",
                "stagnation_index",
                "wind_lag_1",
                "is_fog_time",
            }
        )
    ]
    if any(col in unsafe_visibility_features or "target" in col.lower() for col in features):
        raise ValueError("Visibility target leakage detected in feature columns.")

    X_train = train_df[features]
    X_test = test_df[features]

    model = XGBRegressor(
        n_estimators=1200,
        learning_rate=0.015,
        max_depth=6,
        gamma=2.0,
        reg_lambda=5.0,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        device="cuda",
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    print("      -> Fitting Visibility Specialist (Delta-Sqrt)...")
    model.fit(X_train, y_train_delta, verbose=False)

    preds_delta = model.get_booster().predict(
        xgb.DMatrix(X_test, enable_categorical=True),
        strict_shape=False,
    )
    sqrt_vis_pred = np.sqrt(test_df["vis_lag_1"].to_numpy(dtype=np.float64)) + preds_delta
    preds_abs = np.clip(np.square(sqrt_vis_pred), 150.0, 10000.0)

    r2 = float(r2_score(y_test_abs, preds_abs))
    rmse = float(np.sqrt(mean_squared_error(y_test_abs, preds_abs)))
    mae = float(mean_absolute_error(y_test_abs, preds_abs))
    persistence_r2 = float(r2_score(y_test_abs, test_df["vis_lag_1"]))

    print(f"      -> Vis Specialist metrics: R2 = {r2:.4f} | RMSE = {rmse:.4f} | MAE = {mae:.4f}")
    print(f"      -> Visibility persistence baseline R2: {persistence_r2:.4f}")

    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, "checkpoints/visibility_target_model.joblib")
    return y_test_abs.values, preds_abs
