from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

def _save_bundle_atomic(bundle: dict, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        joblib.dump(bundle, temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _predict_xgb(model, features: pd.DataFrame) -> np.ndarray:
    matrix = xgb.DMatrix(features, enable_categorical=True)
    best_iteration = getattr(model, "best_iteration", None)
    iteration_range = (0, int(best_iteration) + 1) if best_iteration is not None else (0, 0)
    return np.asarray(
        model.get_booster().predict(
            matrix,
            iteration_range=iteration_range,
            strict_shape=False,
        ),
        dtype=np.float64,
    )


def _normalized_inverse_frequency_weights(values: pd.Series, bins: int = 10) -> np.ndarray:
    counts, edges = np.histogram(values.to_numpy(dtype=np.float64), bins=bins)
    indices = np.clip(np.digitize(values, edges) - 1, 0, len(counts) - 1)
    weights = 1.0 / np.maximum(counts[indices], 1)
    return weights / np.mean(weights)


def _fog_persistence_memory(visibility_lag: pd.Series) -> pd.Series:
    fog_event = visibility_lag < 1000.0
    steps_since = np.full(len(visibility_lag), np.nan, dtype=np.float64)
    last_fog_step = None
    for position, is_fog in enumerate(fog_event.fillna(False).to_numpy()):
        if is_fog:
            last_fog_step = position
            steps_since[position] = 0.0
        elif last_fog_step is not None:
            steps_since[position] = position - last_fog_step
    hours_since = steps_since * 0.5
    return pd.Series(np.exp(-hours_since / 4.0), index=visibility_lag.index).fillna(0.0)


def add_vis_features(df: pd.DataFrame) -> pd.DataFrame:
    vis_df = df.copy()
    vis_df["vis_lag_1"] = vis_df["visibility"].shift(1)
    for lag_steps, lag_label in [(6, "3h"), (12, "6h"), (24, "12h"), (48, "24h")]:
        vis_df[f"vis_lag_{lag_label}"] = vis_df["visibility"].shift(lag_steps)
    vis_df["temp_lag_1"] = vis_df["temp"].shift(1)
    vis_df["temp_lag_3"] = vis_df["temp"].shift(3)
    vis_df["wind_lag_1"] = vis_df["wind_speed"].shift(1)
    vis_df["pressure_lag_1"] = vis_df["pressure"].shift(1)

    if "dew_point" in vis_df.columns:
        dew_point_lag = vis_df["dew_point"].shift(1)
    elif "humidity" in vis_df.columns:
        dew_point_lag = vis_df["temp_lag_1"] - (
            (100.0 - vis_df["humidity"].shift(1)) / 5.0
        )
    else:
        dew_point_lag = vis_df["temp_lag_1"] - 5.0

    vis_df["dew_depression"] = vis_df["temp_lag_1"] - dew_point_lag
    vis_df["dew_depression_sq"] = np.square(vis_df["dew_depression"])
    vis_df["dew_depression_velocity_1h"] = vis_df["dew_depression"].diff(2)
    vis_df["dew_depression_velocity_2h"] = vis_df["dew_depression"].diff(4)
    vis_df["vis_velocity_1h"] = vis_df["vis_lag_1"].diff(2)
    vis_df["vis_velocity_2h"] = vis_df["vis_lag_1"].diff(4)
    vis_df["mixing_layer_proxy"] = vis_df["wind_lag_1"] * vis_df["dew_depression"]
    vis_df["fog_onset_signal"] = (
        (vis_df["dew_depression_velocity_1h"] < -0.5)
        & (vis_df["wind_lag_1"] < 5.0)
    ).astype("int8")
    vis_df["vis_roll_mean_3h"] = vis_df["vis_lag_1"].rolling(6).mean()
    vis_df["vis_roll_min_3h"] = vis_df["vis_lag_1"].rolling(6).min()
    vis_df["vis_roll_mean_6h"] = vis_df["vis_lag_1"].rolling(12).mean()
    vis_df["vis_roll_min_6h"] = vis_df["vis_lag_1"].rolling(12).min()
    vis_df["dew_roll_mean_3h"] = vis_df["dew_depression"].rolling(6).mean()
    vis_df["dew_roll_min_3h"] = vis_df["dew_depression"].rolling(6).min()
    vis_df["vis_trend_3h"] = vis_df["vis_lag_3h"] - vis_df["vis_lag_12h"]
    vis_df["fog_deepening"] = (vis_df["vis_trend_3h"] < -500.0).astype("int8")

    if "datetime" in vis_df.columns:
        datetime_values = pd.to_datetime(vis_df["datetime"])
    else:
        datetime_values = vis_df.index
    hour = datetime_values.hour
    month = datetime_values.month
    vis_df["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    vis_df["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    vis_df["nocturnal_fog_score"] = (
        (hour >= 2)
        & (hour <= 6)
        & (vis_df["dew_depression"] < 2.0)
        & (vis_df["wind_lag_1"] < 3.0)
    ).astype("int8")
    vis_df["fog_persistence_memory"] = _fog_persistence_memory(vis_df["vis_lag_1"])

    if {"temp_2m", "temp_surface"}.issubset(vis_df.columns):
        vis_df["boundary_layer_stability"] = (
            vis_df["temp_2m"].shift(1) - vis_df["temp_surface"].shift(1)
        ) / 10.0
    else:
        # Surface temperature is unavailable, so recent cooling is the causal stability proxy.
        vis_df["boundary_layer_stability"] = (
            vis_df["temp_lag_1"] - vis_df["temp_lag_3"]
        ) / 10.0

    vis_df["monsoon_phase"] = np.select(
        [month.isin([4, 5]), month.isin([6, 7, 8, 9]), month.isin([10, 11])],
        [1, 2, 3],
        default=0,
    ).astype("int8")
    return vis_df.dropna().copy()


def _flatline_exclusion_mask(visibility: pd.Series) -> pd.Series:
    rolling_std = visibility.rolling(4).std().fillna(np.inf)
    flatline = rolling_std < 1.0
    flatline_exit = flatline.shift(1, fill_value=False) & ~flatline
    recovery = pd.Series(False, index=visibility.index)
    for offset in range(4):
        recovery |= flatline_exit.shift(offset, fill_value=False)
    return flatline | recovery


def train_and_predict(df_master: pd.DataFrame):
    vis_df = add_vis_features(df_master)
    exclusion_mask = _flatline_exclusion_mask(vis_df["visibility"])
    total_removed = int(exclusion_mask.sum())
    removed_pct = 100.0 * total_removed / len(vis_df)
    vis_df = vis_df.loc[~exclusion_mask].copy()
    print(
        f"      -> Flatline rows removed before split: {total_removed} "
        f"({removed_pct:.2f}%)"
    )

    conditions = [
        vis_df["visibility"] <= 500.0,
        (vis_df["visibility"] > 500.0) & (vis_df["visibility"] <= 2000.0),
        vis_df["visibility"] > 2000.0,
    ]
    vis_df["split_regime"] = np.select(conditions, [0, 1, 2], default=2).astype("int8")

    train_df, test_df = train_test_split(
        vis_df,
        test_size=0.15,
        stratify=vis_df["split_regime"],
        random_state=42,
    )
    train_df = train_df.sort_index()
    test_df = test_df.sort_index()
    if train_df.empty or test_df.empty:
        raise ValueError("Visibility chronological split generated an empty partition.")

    y_train = train_df["visibility"]
    y_test = test_df["visibility"]
    sample_weights = _normalized_inverse_frequency_weights(y_train)
    print(f"      -> Visibility sample-weight mean: {np.mean(sample_weights):.6f}")

    current_visibility_leaks = {
        "visibility_trend",
        "visibility_acceleration",
        "visibility_rolling_mean_3",
        "visibility_rolling_mean_6",
        "visibility_rolling_mean_12",
        "visibility_rolling_std_3",
        "visibility_rolling_std_6",
        "visibility_rolling_std_12",
    }
    drop_cols = [
        column
        for column in vis_df.columns
        if (
            ("vis" in column.lower()
             and "lag" not in column.lower()
             and "velocity" not in column.lower()
             and "trend" not in column.lower()
             and "roll" not in column.lower())
            or column in current_visibility_leaks
            or "target" in column.lower()
            or column
            in {
                "datetime",
                "temp",
                "pressure",
                "humidity",
                "wind_speed",
                "wind_gust",
                "wind_direction",
                "split_regime",
            }
        )
    ]
    features = list(
        dict.fromkeys(
            column
            for column in vis_df.columns
            if column not in drop_cols and pd.api.types.is_numeric_dtype(vis_df[column])
        )
    )
    if "split_regime" in features or "visibility" in features:
        raise ValueError("Visibility split label or target leaked into model features.")

    X_train = train_df[features]
    X_test = test_df[features]
    train_counts = train_df["split_regime"].value_counts().sort_index().to_dict()
    test_counts = test_df["split_regime"].value_counts().sort_index().to_dict()
    print(f"      -> Stratified regime counts: train={train_counts}, test={test_counts}")

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
        random_state=42,
        n_jobs=-1,
    )
    print(f"      -> Fitting Visibility Specialist (Stratified Split, N={len(train_df)})...")
    model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)

    predictions = _predict_xgb(model, X_test)
    predictions = np.clip(predictions, 150.0, 10000.0)
    r2 = float(r2_score(y_test, predictions))
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    mae = float(mean_absolute_error(y_test, predictions))
    print(f"      -> Visibility metrics: R2 = {r2:.4f} | RMSE = {rmse:.4f} | MAE = {mae:.4f}")

    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    _save_bundle_atomic(
        {"model": model, "features": features, "regime_counts": {"train": train_counts, "test": test_counts}},
        Path("checkpoints/visibility_target_model.joblib"),
    )
    return y_test.values, predictions
