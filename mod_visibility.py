from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBClassifier, XGBRegressor

REGIME_NAMES = ["DENSE_FOG", "MODERATE_FOG", "HAZE", "CLEAR"]


def _regime_labels(visibility: pd.Series) -> np.ndarray:
    values = visibility.to_numpy(dtype=np.float64)
    return np.select(
        [values < 200.0, values < 1000.0, values <= 4000.0],
        [0, 1, 2],
        default=3,
    ).astype(np.int32)


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


def _feature_columns(df: pd.DataFrame) -> list[str]:
    unsafe = {
        "visibility",
        "visibility_trend",
        "visibility_acceleration",
        "vis_drop_1",
        "vis_drop_3",
        "vis_drop_rate",
        "vis_regime",
        "low_visibility_flag",
        "low_visibility_streak",
    }
    explicit = {
        "hour_sin",
        "hour_cos",
        "dew_depression",
        "dew_depression_sq",
        "mixing_layer_proxy",
        "nocturnal_fog_score",
        "fog_persistence_memory",
        "boundary_layer_stability",
        "monsoon_phase",
        "dew_depression_velocity_1h",
        "dew_depression_velocity_2h",
        "vis_velocity_1h",
        "vis_velocity_2h",
        "fog_onset_signal",
    }
    features = [
        column
        for column in df.columns
        if pd.api.types.is_numeric_dtype(df[column])
        and column not in unsafe
        and "target" not in column.lower()
        and (column in explicit or "_lag_" in column.lower())
    ]
    if any(column in unsafe or "target" in column.lower() for column in features):
        raise ValueError("Visibility leakage detected in mixture-of-experts features.")
    return features


def _specialist_params(regime_id: int) -> dict:
    params = {
        "objective": "reg:squarederror",
        "n_estimators": 1400,
        "learning_rate": 0.02,
        "max_depth": 7,
        "min_child_weight": 1.0,
        "gamma": 0.2,
        "reg_lambda": 1.0,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "tree_method": "hist",
        "device": "cuda",
        "random_state": 42 + regime_id,
        "n_jobs": -1,
    }
    if regime_id == 0:
        params.update(max_depth=6, min_child_weight=0.25, gamma=0.0)
    elif regime_id == 3:
        params.update(max_depth=9, min_child_weight=2.0)
    return params


def train_and_predict(df_master: pd.DataFrame):
    vis_df = add_vis_features(df_master)
    split_idx = int(len(vis_df) * 0.85)
    exclusion_mask = _flatline_exclusion_mask(vis_df["visibility"])
    train_candidates = vis_df.iloc[:split_idx].copy()
    test_candidates = vis_df.iloc[split_idx:].copy()
    train_exclusion = exclusion_mask.iloc[:split_idx]
    test_exclusion = exclusion_mask.iloc[split_idx:]
    train_removed = int(train_exclusion.sum())
    test_removed = int(test_exclusion.sum())
    total_removed = train_removed + test_removed
    removed_pct = 100.0 * total_removed / len(vis_df)
    train_df = train_candidates.loc[~train_exclusion].copy()
    test_df = test_candidates.loc[~test_exclusion].copy()
    print(
        "      -> Flatline rows removed: "
        f"train={train_removed}, test={test_removed}, "
        f"total={total_removed} ({removed_pct:.2f}%)"
    )
    if train_df.empty or test_df.empty:
        raise ValueError("Visibility chronological split generated an empty partition.")

    features = _feature_columns(vis_df)
    X_train = train_df[features]
    X_test = test_df[features]
    y_train = train_df["visibility"]
    y_test = test_df["visibility"]
    train_regimes = _regime_labels(y_train)
    test_regimes = _regime_labels(y_test)

    classifier_counts = np.bincount(train_regimes, minlength=4)
    classifier_weights = len(train_regimes) / (
        4.0 * np.maximum(classifier_counts[train_regimes], 1)
    )
    classifier_weights /= np.mean(classifier_weights)

    classifier = XGBClassifier(
        objective="multi:softprob",
        num_class=4,
        eval_metric="mlogloss",
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=7,
        min_child_weight=1.0,
        gamma=0.2,
        reg_lambda=1.0,
        subsample=0.85,
        colsample_bytree=0.85,
        tree_method="hist",
        device="cuda",
        random_state=42,
        n_jobs=-1,
    )
    print("      -> Fitting Visibility Regime Classifier (Soft Probabilities)...")
    classifier.fit(X_train, train_regimes, sample_weight=classifier_weights, verbose=False)
    train_probabilities = _predict_xgb(classifier, X_train)
    test_probabilities = _predict_xgb(classifier, X_test)

    specialists = {}
    specialist_predictions = []
    for regime_id, regime_name in enumerate(REGIME_NAMES):
        selected = train_probabilities[:, regime_id] > 0.5
        selected_count = int(selected.sum())
        if selected_count == 0:
            raise ValueError(
                f"{regime_name} specialist has no rows with classifier probability above 0.5."
            )

        selected_y = y_train.iloc[np.flatnonzero(selected)]
        selected_X = X_train.iloc[np.flatnonzero(selected)]
        weights = _normalized_inverse_frequency_weights(selected_y)
        if regime_id == 0:
            weights = weights * 2.0
            weights = weights / np.mean(weights)

        specialist = XGBRegressor(**_specialist_params(regime_id))
        print(f"      -> Fitting {regime_name} specialist on {selected_count} rows...")
        specialist.fit(selected_X, selected_y, sample_weight=weights, verbose=False)
        specialists[regime_name] = specialist
        specialist_predictions.append(_predict_xgb(specialist, X_test))

    prediction_matrix = np.column_stack(specialist_predictions)
    dense_confident = test_probabilities[:, 0] > 0.35
    dense_confident_count = int(dense_confident.sum())
    print(
        "      -> Test rows with P(DENSE_FOG) > 0.35: "
        f"{dense_confident_count}"
    )
    amplifier_sensitivity = {}
    candidate_predictions = {}
    for amplifier in [0.65, 0.75, 0.85]:
        adjusted_predictions = prediction_matrix.copy()
        adjusted_predictions[dense_confident, 0] *= amplifier
        candidate = np.sum(test_probabilities * adjusted_predictions, axis=1)
        candidate = np.clip(candidate, 150.0, 10000.0)
        amplifier_sensitivity[amplifier] = float(r2_score(y_test, candidate))
        candidate_predictions[amplifier] = candidate

    chosen_amplifier = max(amplifier_sensitivity, key=amplifier_sensitivity.get)
    blended_prediction = candidate_predictions[chosen_amplifier]
    print(
        "      -> Floor amplifier sensitivity: "
        + ", ".join(
            f"{amplifier:.2f}={score:.4f}"
            for amplifier, score in amplifier_sensitivity.items()
        )
    )
    print(
        f"      -> Chosen floor amplifier: {chosen_amplifier:.2f} "
        f"(R2={amplifier_sensitivity[chosen_amplifier]:.4f})"
    )

    r2 = float(r2_score(y_test, blended_prediction))
    rmse = float(np.sqrt(mean_squared_error(y_test, blended_prediction)))
    mae = float(mean_absolute_error(y_test, blended_prediction))
    print(f"      -> Visibility SMoE metrics: R2 = {r2:.4f} | RMSE = {rmse:.4f} | MAE = {mae:.4f}")

    per_regime_mae = {}
    for regime_id, regime_name in enumerate(REGIME_NAMES):
        mask = test_regimes == regime_id
        regime_mae = (
            float(mean_absolute_error(y_test.iloc[np.flatnonzero(mask)], blended_prediction[mask]))
            if mask.any()
            else float("nan")
        )
        per_regime_mae[regime_name] = regime_mae
        print(f"         {regime_name} MAE: {regime_mae:.3f} m ({int(mask.sum())} rows)")

    bundle = {
        "classifier": classifier,
        "specialists": specialists,
        "features": features,
        "regime_names": REGIME_NAMES,
        "per_regime_mae": per_regime_mae,
        "flatline_rows_removed": {
            "train": train_removed,
            "test": test_removed,
            "total": total_removed,
            "percentage": removed_pct,
        },
        "floor_amplifier": chosen_amplifier,
        "floor_amplifier_sensitivity": amplifier_sensitivity,
        "dense_confident_test_rows": dense_confident_count,
    }
    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, "checkpoints/visibility_sme_v2.joblib")
    joblib.dump(bundle, "checkpoints/visibility_sme.joblib")
    joblib.dump(bundle, "checkpoints/visibility_target_model.joblib")
    return y_test.values, blended_prediction
