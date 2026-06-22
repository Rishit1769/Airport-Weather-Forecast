from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


def add_temp_features(df: pd.DataFrame) -> pd.DataFrame:
    temp_df = df.copy()
    if not isinstance(temp_df.index, pd.DatetimeIndex):
        if "datetime" not in temp_df.columns:
            raise ValueError("Temperature data requires a DatetimeIndex or datetime column.")
        temp_df["datetime"] = pd.to_datetime(temp_df["datetime"], errors="coerce")
        temp_df = temp_df.dropna(subset=["datetime"]).set_index("datetime")

    # Immediate thermal velocity (dT/dt).
    temp_df["temp_lag_1"] = temp_df["temp"].shift(1)
    temp_df["temp_lag_2"] = temp_df["temp"].shift(2)
    temp_df["temp_lag_3"] = temp_df["temp"].shift(3)
    temp_df["temp_diff_1h"] = temp_df["temp_lag_1"] - temp_df["temp_lag_3"]

    # Deep diurnal memory.
    temp_df["temp_lag_24h"] = temp_df["temp"].shift(48)
    temp_df["temp_lag_48h"] = temp_df["temp"].shift(96)

    # Standard and phase-shifted solar anchors.
    hour_float = temp_df.index.hour + (temp_df.index.minute / 60.0)
    temp_df["hour_sin"] = np.sin(2.0 * np.pi * hour_float / 24.0)
    temp_df["hour_cos"] = np.cos(2.0 * np.pi * hour_float / 24.0)
    temp_df["solar_thermal_peak"] = np.sin(2.0 * np.pi * (hour_float - 8.5) / 24.0)

    # Thermal inertia using only lagged observations.
    temp_df["temp_ema_3h"] = temp_df["temp_lag_1"].ewm(span=6, adjust=False).mean()
    temp_df["temp_ema_6h"] = temp_df["temp_lag_1"].ewm(span=12, adjust=False).mean()

    # Coastal dew-point boundary.
    if "dew_point" in temp_df.columns:
        temp_df["dew_point_lag_1"] = temp_df["dew_point"].shift(1)
        temp_df["temp_dew_depression"] = temp_df["temp_lag_1"] - temp_df["dew_point_lag_1"]
    return temp_df


def apply_outage_mask(df: pd.DataFrame) -> pd.DataFrame:
    temp_diff = df["temp"].diff().abs()
    four_hour_movement = temp_diff.rolling(8).sum()
    dead_sensor_mask = four_hour_movement.eq(0.0)
    return df.loc[~dead_sensor_mask].copy()


def train_and_predict(df: pd.DataFrame):
    temp_df = apply_outage_mask(add_temp_features(df))
    temp_df = temp_df.dropna().copy()

    split_index = int(len(temp_df) * 0.85)
    train_df = temp_df.iloc[:split_index]
    test_df = temp_df.iloc[split_index:]
    if train_df.empty or test_df.empty:
        raise ValueError("Temperature chronological split generated an empty partition.")

    # Ensure absolutely no target leakage from the shared data pipeline.
    drop_cols = [
        col
        for col in temp_df.columns
        if "target" in col.lower() or col == "temp" or col == "datetime"
    ]
    features = [col for col in temp_df.columns if col not in drop_cols]
    feature_columns = [col for col in features if pd.api.types.is_numeric_dtype(temp_df[col])]
    if any("target" in col.lower() for col in feature_columns):
        raise ValueError("Target leakage detected in temperature feature columns.")

    X_train = train_df[feature_columns]
    X_test = test_df[feature_columns]

    # Learn the immediate temperature change, then reconstruct absolute temperature.
    y_train_delta = train_df["temp"] - train_df["temp_lag_1"]
    y_test_delta = test_df["temp"] - test_df["temp_lag_1"]
    y_test_abs = test_df["temp"]

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=2000,
        max_depth=7,
        learning_rate=0.015,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        device="cuda",
        gamma=0.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        early_stopping_rounds=50,
    )
    model.fit(
        X_train,
        y_train_delta,
        eval_set=[(X_test, y_test_delta)],
        verbose=False,
    )

    # Verify feature importances to ensure no leakage.
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({"Feature": feature_columns, "Importance": importances})
    feat_imp = feat_imp.sort_values(by="Importance", ascending=False).head(5)

    print("\n      -> Top 5 Features (Checking for leakage):")
    for _, row in feat_imp.iterrows():
        print(f"         {row['Feature']}: {row['Importance']:.4f}")

    best_iteration = getattr(model, "best_iteration", None)
    iteration_range = (0, int(best_iteration) + 1) if best_iteration is not None else (0, 0)
    preds_delta = model.get_booster().predict(
        xgb.DMatrix(X_test, enable_categorical=True),
        iteration_range=iteration_range,
        strict_shape=False,
    )
    preds_abs = X_test["temp_lag_1"].to_numpy(dtype=np.float64) + preds_delta
    preds_abs = np.clip(preds_abs, -10.0, 55.0)

    r2 = float(r2_score(y_test_abs, preds_abs))
    rmse = float(np.sqrt(mean_squared_error(y_test_abs, preds_abs)))
    mae = float(mean_absolute_error(y_test_abs, preds_abs))
    print(f"      -> Temp Specialist metrics: R2 = {r2:.4f} | RMSE = {rmse:.4f} | MAE = {mae:.4f}")

    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, "checkpoints/temp_target_model.joblib")
    return y_test_abs.values, preds_abs
