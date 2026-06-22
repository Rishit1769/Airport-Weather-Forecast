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

    if "wind_speed" in vis_df.columns:
        vis_df["wind_lag_1"] = vis_df["wind_speed"].shift(1)
        stagnation = 1.0 / (vis_df["wind_lag_1"] + 1.0)
        vis_df["stagnation_24h"] = stagnation.rolling(48).sum().fillna(0.0)

    return vis_df.dropna(subset=["temp_lag_1"])


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

    y_train = train_df["visibility"]
    y_test_abs = test_df["visibility"]

    counts, bins = np.histogram(y_train, bins=10)
    bin_indices = np.digitize(y_train, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(counts) - 1)
    safe_counts = np.maximum(counts, 1)
    sample_weights = 1.0 / safe_counts[bin_indices]
    sample_weights = sample_weights / np.mean(sample_weights)
    print(f"      -> Visibility sample-weight mean: {np.mean(sample_weights):.6f}")

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
                "stagnation_24h",
                "wind_lag_1",
            }
        )
    ]
    if any(col in unsafe_visibility_features or "target" in col.lower() for col in features):
        raise ValueError("Visibility target leakage detected in feature columns.")

    X_train = train_df[features]
    X_test = test_df[features]

    constraints = {}
    if "dew_depression" in features:
        constraints["dew_depression"] = 1
    if "stagnation_24h" in features:
        constraints["stagnation_24h"] = -1
    if "wind_lag_1" in features:
        constraints["wind_lag_1"] = 1
    monotone_tuple = tuple(constraints.get(feature, 0) for feature in features)

    model = XGBRegressor(
        n_estimators=2000,
        learning_rate=0.015,
        max_depth=8,
        gamma=0.5,
        subsample=0.85,
        colsample_bytree=0.85,
        tree_method="hist",
        device="cuda",
        objective="reg:squarederror",
        monotone_constraints=monotone_tuple,
        random_state=42,
        n_jobs=-1,
    )

    print("      -> Fitting Visibility Specialist (Linear Space + Weights + Constraints)...")
    model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)

    preds = model.get_booster().predict(
        xgb.DMatrix(X_test, enable_categorical=True),
        strict_shape=False,
    )
    preds_abs = np.clip(preds, 150.0, 10000.0)

    r2 = float(r2_score(y_test_abs, preds_abs))
    rmse = float(np.sqrt(mean_squared_error(y_test_abs, preds_abs)))
    mae = float(mean_absolute_error(y_test_abs, preds_abs))
    persistence_r2 = float(r2_score(y_test_abs, test_df["vis_lag_1"]))

    print(f"      -> Vis Specialist metrics: R2 = {r2:.4f} | RMSE = {rmse:.4f} | MAE = {mae:.4f}")
    print(f"      -> Visibility persistence baseline R2: {persistence_r2:.4f}")

    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, "checkpoints/visibility_target_model.joblib")
    return y_test_abs.values, preds_abs
