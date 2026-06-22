import argparse
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

SEED = 42
FREQ = "30min"
HORIZON = 12  # 6 hours at 30-minute intervals
LAG_STEPS = [1, 2, 3, 6, 12]
ROLLING_MEAN_WINDOWS = [3, 6]
ROLLING_STD_WINDOWS = [3, 6, 12]

TRAIN_END = "2024-01-01"
VAL_END = "2025-01-01"
TEST_END = "2026-01-01"

BASE_CONTINUOUS = [
    "wind_dir",
    "wind_speed",
    "visibility",
    "temp",
    "dew_point",
    "humidity",
    "pressure",
    "cloud_cover",
]
BASE_BINARY = ["is_rain", "is_fog", "is_haze"]

PHYSICAL_BOUNDS = {
    "wind_dir": (0.0, 360.0),
    "wind_speed": (0.0, 80.0),
    "visibility": (0.0, 12000.0),
    "temp": (-10.0, 55.0),
    "dew_point": (-20.0, 40.0),
    "humidity": (0.0, 100.0),
    "pressure": (950.0, 1050.0),
    "cloud_cover": (0.0, 8.0),
}

CORE_TARGETS = ["temp", "wind_speed", "visibility", "pressure", "humidity"]
TARGET_COLUMNS = [f"{c}_target" for c in CORE_TARGETS]

PRED_CLIP_BOUNDS = {
    "temp_target": (-10.0, 55.0),
    "wind_speed_target": (0.0, 80.0),
    "visibility_target": (0.0, 12000.0),
    "pressure_target": (950.0, 1050.0),
    "humidity_target": (0.0, 100.0),
}


@dataclass
class DataSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


@dataclass
class StageConfig:
    name: str
    add_enhanced_signals: bool
    add_event_probability_feature: bool
    use_visibility_weighting: bool
    reg_params: Dict[str, float]


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    random.seed(seed)


def timed_step(name: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            logger.info(f"START: {name}")
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            logger.info(f"END: {name} | Took {elapsed:.2f}s")
            return result
        return wrapper
    return decorator


def import_xgboost():
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError("xgboost is required. Install with: pip install xgboost") from exc
    return xgb


def get_gpu_runtime_params(xgb_module) -> Dict[str, object]:
    params = {
        "tree_method": "hist",
        "device": "cuda",
    }
    probe = xgb_module.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1,
        max_depth=1,
        learning_rate=0.3,
        tree_method=params["tree_method"],
        device=params["device"],
    )
    Xp = pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, 0.0]})
    yp = pd.Series([0.0, 1.0])
    probe.fit(Xp, yp, verbose=False)
    return params


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime"]).set_index("datetime")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Expected a DatetimeIndex or a datetime column.")
    return df.sort_index()


def encode_weather_codes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "weather_codes" not in df.columns:
        return df

    codes = df["weather_codes"].fillna("").astype(str).str.upper()
    df["weather_codes"] = codes
    patterns = {
        "is_haze_code": r"\bHZ\b",
        "is_mist": r"\bBR\b",
        "is_smoke": r"\bFU\b",
        "is_rain_code": r"\b(?:RA|DZ)\b",
        "is_thunderstorm": r"\b(?:TSRA|TS)\b",
    }
    for col, pattern in patterns.items():
        df[col] = codes.str.contains(pattern, regex=True, na=False).astype("int8")

    df["is_haze"] = df.get("is_haze", 0)
    df["is_haze"] = pd.to_numeric(df["is_haze"], errors="coerce").fillna(0).astype("int8")
    df["is_haze"] = np.maximum(df["is_haze"].to_numpy(dtype=np.int8), df["is_haze_code"].to_numpy(dtype=np.int8))
    df["is_haze"] = df["is_haze"].astype("int8")

    df["is_rain"] = df.get("is_rain", 0)
    df["is_rain"] = pd.to_numeric(df["is_rain"], errors="coerce").fillna(0).astype("int8")
    df["is_rain"] = np.maximum(df["is_rain"].to_numpy(dtype=np.int8), df["is_rain_code"].to_numpy(dtype=np.int8))
    df["is_rain"] = df["is_rain"].astype("int8")

    return df


def apply_wind_speed_limits(df: pd.DataFrame, percentile_threshold: float = None) -> pd.DataFrame:
    df = df.copy()
    if "wind_speed" not in df.columns:
        return df
    df["wind_speed"] = pd.to_numeric(df["wind_speed"], errors="coerce")
    df["wind_speed"] = df["wind_speed"].clip(upper=45.0)
    if percentile_threshold is not None and np.isfinite(percentile_threshold):
        df["wind_speed"] = df["wind_speed"].clip(upper=float(min(45.0, percentile_threshold)))
    return df


def calibrate_wind_speed_threshold(clean_df: pd.DataFrame) -> float:
    train_start = pd.Timestamp("2016-01-01")
    train_end = pd.Timestamp(TRAIN_END)
    train_df = clean_df.loc[(clean_df.index >= train_start) & (clean_df.index < train_end)]
    source = train_df if not train_df.empty else clean_df
    threshold = float(np.nanpercentile(source["wind_speed"].to_numpy(dtype=np.float64), 99.9))
    threshold = min(45.0, threshold)
    logger.info(f"Wind-speed 99.9th percentile threshold (train set): {threshold:.3f}")
    return threshold


def validate_no_nan_inf(df: pd.DataFrame, columns: List[str], label: str) -> None:
    arr = df[columns].to_numpy(dtype=np.float64)
    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())
    if nan_count > 0 or inf_count > 0:
        raise ValueError(f"{label} has invalid values: NaN={nan_count}, Inf={inf_count}")


def visibility_distribution(df: pd.DataFrame) -> Dict[str, int]:
    visibility = pd.to_numeric(df["visibility"], errors="coerce")
    return {
        "severe_<1000": int((visibility < 1000.0).sum()),
        "low_1000_5000": int(((visibility >= 1000.0) & (visibility < 5000.0)).sum()),
        "normal_>=5000": int((visibility >= 5000.0).sum()),
    }


@timed_step("load_and_clean")
def load_and_clean(input_csv: str) -> pd.DataFrame:
    raw_df = pd.read_csv(input_csv)
    raw_rows = len(raw_df)
    df = _ensure_datetime_index(raw_df)
    cleaned_rows = len(df)
    dropped_on_datetime = raw_rows - cleaned_rows

    full_index = pd.date_range(df.index.min(), df.index.max(), freq=FREQ)
    df = df.reindex(full_index)

    interpolated_values = 0

    if "pressure" in df.columns:
        df["pressure"] = pd.to_numeric(df["pressure"], errors="coerce")
        pressure_jump_mask = df["pressure"].diff().abs() > 10.0
        df.loc[pressure_jump_mask, "pressure"] = np.nan

    if "wind_speed" in df.columns:
        df["wind_speed"] = pd.to_numeric(df["wind_speed"], errors="coerce")
        wind_jump_mask = df["wind_speed"].diff().abs() > 30.0
        df.loc[wind_jump_mask, "wind_speed"] = np.nan
        df["wind_speed"] = df["wind_speed"].clip(upper=45.0)

    for col in BASE_CONTINUOUS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            lo, hi = PHYSICAL_BOUNDS[col]
            df[col] = df[col].clip(lo, hi)
            nan_before = int(df[col].isna().sum())
            df[col] = df[col].interpolate(method="time", limit=2, limit_direction="both", limit_area="inside")
            nan_after = int(df[col].isna().sum())
            interpolated_values += max(0, nan_before - nan_after)

    for col in BASE_BINARY:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].ffill(limit=2).bfill(limit=2).clip(0, 1)

    df = df.ffill(limit=4).bfill(limit=2)
    df = encode_weather_codes(df)

    if "wind_speed" in df.columns:
        df["wind_speed"] = pd.to_numeric(df["wind_speed"], errors="coerce").clip(upper=45.0)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        df[numeric_cols] = df[numeric_cols].ffill().bfill()

    logger.info(f"Rows dropped due to invalid datetime: {dropped_on_datetime}")
    logger.info(f"Estimated values interpolated (continuous): {interpolated_values}")
    for col in ["temp", "wind_speed", "visibility", "pressure", "humidity"]:
        if col in df.columns:
            logger.info(f"Sanity {col}: min={float(df[col].min()):.3f}, max={float(df[col].max()):.3f}")
    logger.info(f"NaNs after load_and_clean: {int(df.isna().sum().sum())}")

    # Derive engineered signals for add_features optional pickup
    df["temp_dew_diff"] = (df["temp"] - df["dew_point"]).clip(-30.0, 60.0)
    df["pressure_change"] = df["pressure"].diff().clip(-20.0, 20.0).fillna(0.0)
    df["wind_speed_change"] = df["wind_speed"].diff().clip(-40.0, 40.0).fillna(0.0)

    return df


@timed_step("add_features")
def add_features(
    df: pd.DataFrame,
    add_enhanced_signals: bool = False,
) -> pd.DataFrame:
    df = _ensure_datetime_index(df).copy()

    df["wind_dir_sin"] = np.sin(np.radians(df["wind_dir"]))
    df["wind_dir_cos"] = np.cos(np.radians(df["wind_dir"]))
    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12.0)

    # Only include columns that are guaranteed to exist
    stable_cols = [
        "temp",
        "wind_speed",
        "visibility",
        "pressure",
        "humidity",
        "wind_dir_sin",
        "wind_dir_cos",
    ]

    # Add optional columns only if they exist in the dataframe
    for optional_col in [
        "temp_dew_diff",
        "pressure_change",
        "wind_speed_change",
        "is_haze_code",
        "is_mist",
        "is_smoke",
        "is_rain_code",
        "is_thunderstorm",
    ]:
        if optional_col in df.columns:
            stable_cols.append(optional_col)

    # Core interactions
    df["humidity_wind"] = (df["humidity"] * df["wind_speed"]).clip(0.0, 8000.0)
    df["pressure_humidity"] = (df["pressure"] * df["humidity"]).clip(0.0, 120000.0)
    df["temp_humidity"] = (df["temp"] * df["humidity"]).clip(-1000.0, 6000.0)
    df["humidity_temperature"] = (df["humidity"] * df["temp"]).clip(-1000.0, 6000.0)

    vis_trend = df["visibility"] - df["visibility"].shift(1)
    df["visibility_trend"] = vis_trend.clip(-5000.0, 5000.0).astype("float32")
    vis_acc = vis_trend - vis_trend.shift(1)
    df["visibility_acceleration"] = vis_acc.clip(-5000.0, 5000.0).astype("float32")

    # Build all new columns in a dict first to avoid fragmentation
    new_cols: Dict[str, pd.Series] = {}

    for col in stable_cols:
        for step in LAG_STEPS:
            new_cols[f"{col}_lag_{step}"] = df[col].shift(step)
        for window in ROLLING_MEAN_WINDOWS:
            new_cols[f"{col}_rolling_mean_{window}"] = df[col].rolling(window).mean()
        for window in ROLLING_STD_WINDOWS:
            new_cols[f"{col}_rolling_std_{window}"] = df[col].rolling(window).std()

    # Explicit visibility history features
    new_cols["visibility_lag_1"] = df["visibility"].shift(1)
    new_cols["visibility_lag_3"] = df["visibility"].shift(3)
    new_cols["visibility_lag_6"] = df["visibility"].shift(6)
    new_cols["visibility_rolling_mean_3"] = df["visibility"].rolling(3).mean()
    new_cols["visibility_rolling_mean_6"] = df["visibility"].rolling(6).mean()
    new_cols["visibility_rolling_mean_12"] = df["visibility"].rolling(12).mean()
    new_cols["visibility_rolling_std_3"] = df["visibility"].rolling(3).std()
    new_cols["visibility_rolling_std_6"] = df["visibility"].rolling(6).std()
    new_cols["visibility_rolling_std_12"] = df["visibility"].rolling(12).std()

    vis_lag_1 = df["visibility"].shift(1)
    vis_lag_3 = df["visibility"].shift(3)
    vis_lag_6 = df["visibility"].shift(6)
    new_cols["wind_speed_x_visibility_lag_1"] = df["wind_speed"] * vis_lag_1
    new_cols["wind_speed_x_visibility_lag_3"] = df["wind_speed"] * vis_lag_3
    new_cols["wind_speed_x_visibility_lag_6"] = df["wind_speed"] * vis_lag_6

    new_cols["vis_drop_1"] = df["visibility"] - vis_lag_1
    new_cols["vis_drop_3"] = df["visibility"] - vis_lag_3
    new_cols["vis_drop_rate"] = new_cols["vis_drop_1"] / (vis_lag_1 + 1.0)
    new_cols["high_humidity_flag"] = (df["humidity"] > 90.0).astype("float32")
    new_cols["humidity_spike"] = df["humidity"] - df["humidity"].shift(3)
    new_cols["low_wind_flag"] = (df["wind_speed"] < 2.0).astype("float32")

    if "dew_point" in df.columns:
        dew_gap = np.abs(df["temp"] - df["dew_point"])
        new_cols["dew_gap"] = dew_gap
        new_cols["dew_gap_lag_3"] = dew_gap.shift(3)
        new_cols["dew_gap_change"] = dew_gap - dew_gap.shift(3)

    new_cols["pressure_drop_fast"] = df["pressure"] - df["pressure"].shift(3)

    # Concat all new columns at once to avoid fragmentation
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    if add_enhanced_signals:
        hour = df.index.hour
        enhanced_cols: Dict[str, pd.Series] = {}

        if "dew_point" in df.columns:
            dew_proximity = np.abs(df["temp"] - df["dew_point"])
            enhanced_cols["dew_proximity"] = dew_proximity.clip(0.0, 30.0).astype("float32")
            enhanced_cols["near_dew_flag"] = (dew_proximity < 2.0).astype("float32")

        vis_bins = [0.0, 1000.0, 3000.0, 8000.0, 12000.0]
        enhanced_cols["vis_regime"] = pd.cut(
            df["visibility"],
            bins=vis_bins,
            labels=[0, 1, 2, 3],
            include_lowest=True,
            right=False,
        ).astype("float32")

        low_vis = (df["visibility"] < 3000.0).astype("int32")
        enhanced_cols["low_visibility_flag"] = low_vis.astype("float32")
        group = (low_vis == 0).cumsum()
        streak = low_vis.groupby(group).cumsum().astype("float32")
        enhanced_cols["low_visibility_streak"] = streak.clip(0.0, 96.0)

        enhanced_cols["pressure_change_3h"] = (df["pressure"] - df["pressure"].shift(6)).clip(-20.0, 20.0).astype("float32")

        morning_flag = ((hour >= 4) & (hour <= 8)).astype("float32")
        enhanced_cols["morning_humidity"] = (df["humidity"] * morning_flag).clip(0.0, 100.0).astype("float32")

        df = pd.concat([df, pd.DataFrame(enhanced_cols, index=df.index)], axis=1)

    if "pressure_change" in df.columns:
        df = df[df["pressure_change"].abs() < 20.0]
    if "wind_speed_change" in df.columns:
        df = df[df["wind_speed_change"].abs() < 40.0]

    df = df.dropna().copy()

    float_cols = df.select_dtypes(include=["float64"]).columns
    if len(float_cols) > 0:
        df[float_cols] = df[float_cols].astype("float32")

    return df


@timed_step("add_targets")
def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_datetime_index(df).copy()
    for col in CORE_TARGETS:
        df[f"{col}_target"] = df[col].shift(-HORIZON)
    df = df.dropna().copy()

    feature_cols = get_feature_columns(df)
    validate_no_nan_inf(df, feature_cols, "Features")
    validate_no_nan_inf(df, TARGET_COLUMNS, "Targets")

    logger.info("Target ranges before training:")
    for t in TARGET_COLUMNS:
        logger.info(f"{t}: min={float(df[t].min()):.3f}, max={float(df[t].max()):.3f}")

    return df


def split_chronological(df: pd.DataFrame) -> DataSplits:
    train = df.loc[(df.index >= "2016-01-01") & (df.index < TRAIN_END)].copy()
    val = df.loc[(df.index >= TRAIN_END) & (df.index < VAL_END)].copy()
    test = df.loc[(df.index >= "2025-01-01") & (df.index < "2026-01-01")].copy()
    if train.empty or val.empty or test.empty:
        raise ValueError("Chronological split generated empty partition(s).")
    return DataSplits(train=train, val=val, test=test)


def get_feature_columns(df: pd.DataFrame, missing_reference_df: pd.DataFrame = None) -> List[str]:
    cols = [c for c in df.columns if not c.endswith("_target")]
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    ref_df = missing_reference_df if missing_reference_df is not None else df

    selected: List[str] = []
    for c in numeric_cols:
        missing_ratio = float(ref_df[c].isna().mean()) if c in ref_df.columns else 0.0
        if missing_ratio > 0.30:
            continue
        std_val = float(df[c].std(ddof=0))
        if not np.isfinite(std_val) or std_val < 1e-6:
            continue
        selected.append(c)

    return selected


def build_visibility_sample_weights(y_train: pd.Series) -> Tuple[np.ndarray, Dict[str, float]]:
    y_arr = y_train.to_numpy(dtype=np.float64)
    weights = np.ones_like(y_arr, dtype=np.float64)
    low_mask = (y_arr >= 1000.0) & (y_arr < 3000.0)
    severe_mask = y_arr < 1000.0
    weights[low_mask] = 6.0
    weights[severe_mask] = 15.0
    stats = {
        "moderate_low_count": int(low_mask.sum()),
        "low_visibility_count": int(low_mask.sum()),
        "severe_visibility_count": int(severe_mask.sum()),
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "weight_mean": float(weights.mean()),
    }
    return weights, stats


@timed_step("train_model")
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    runtime: Dict[str, object],
    reg_params: Dict[str, float],
    sample_weight_train: np.ndarray = None,
):
    xgb = import_xgboost()

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
        gamma=reg_params.get("gamma", 0.0),
        reg_alpha=reg_params.get("reg_alpha", 0.0),
        reg_lambda=reg_params.get("reg_lambda", 1.0),
    )
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def clip_prediction(target_col: str, pred: np.ndarray) -> np.ndarray:
    lo, hi = PRED_CLIP_BOUNDS[target_col]
    return np.clip(pred, lo, hi)


def validate_prediction_stability(target_col: str, raw_pred: np.ndarray, clipped_pred: np.ndarray) -> None:
    if np.isnan(raw_pred).any() or np.isinf(raw_pred).any():
        raise ValueError(f"Prediction instability for {target_col}: NaN/Inf in raw output.")
    if np.isnan(clipped_pred).any() or np.isinf(clipped_pred).any():
        raise ValueError(f"Prediction instability for {target_col}: NaN/Inf after clipping.")
    lo, hi = PRED_CLIP_BOUNDS[target_col]
    if (clipped_pred < lo).any() or (clipped_pred > hi).any():
        raise ValueError(f"Prediction instability for {target_col}: values outside physical bounds.")


@timed_step("predict")
def predict(models: Dict[str, object], X: pd.DataFrame) -> Dict[str, np.ndarray]:
    preds: Dict[str, np.ndarray] = {}
    for target_col, model in models.items():
        raw = np.asarray(model.predict(X), dtype=np.float64)
        clipped = clip_prediction(target_col, raw)
        validate_prediction_stability(target_col, raw, clipped)
        preds[target_col] = clipped
    return preds


@timed_step("evaluate")
def evaluate(y_df: pd.DataFrame, preds: Dict[str, np.ndarray]) -> Dict[str, object]:
    metrics: Dict[str, object] = {}
    for target_col in TARGET_COLUMNS:
        y_true = y_df[target_col].to_numpy(dtype=np.float64)
        y_pred = preds[target_col]
        metrics[target_col] = {
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }
    visibility_true = y_df["visibility_target"].to_numpy(dtype=np.float64)
    visibility_pred = preds["visibility_target"]
    low_mask = visibility_true < 5000.0
    severe_mask = visibility_true < 1000.0
    metrics["visibility_summary"] = {
        "overall_r2": float(metrics["visibility_target"]["r2"]),
        "low_visibility_r2": float(r2_score(visibility_true[low_mask], visibility_pred[low_mask])) if int(low_mask.sum()) > 1 else float("nan"),
        "severe_visibility_r2": float(r2_score(visibility_true[severe_mask], visibility_pred[severe_mask])) if int(severe_mask.sum()) > 1 else float("nan"),
        "visibility_mae": float(mean_absolute_error(visibility_true, visibility_pred)),
        "low_visibility_mae": float(mean_absolute_error(visibility_true[low_mask], visibility_pred[low_mask])) if int(low_mask.sum()) > 0 else float("nan"),
        "severe_visibility_mae": float(mean_absolute_error(visibility_true[severe_mask], visibility_pred[severe_mask])) if int(severe_mask.sum()) > 0 else float("nan"),
        "low_visibility_count": int(low_mask.sum()),
        "severe_visibility_count": int(severe_mask.sum()),
    }
    return metrics


def evaluate_segments(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    segments = {
        "severe": y_true < 1000.0,
        "moderate": (y_true >= 1000.0) & (y_true < 3000.0),
        "clear": y_true >= 3000.0,
    }
    out: Dict[str, Dict[str, float]] = {}
    for name, mask in segments.items():
        if int(mask.sum()) == 0:
            out[name] = {"count": 0, "rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
            continue
        out[name] = {
            "count": int(mask.sum()),
            "rmse": float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))),
            "mae": float(mean_absolute_error(y_true[mask], y_pred[mask])),
            "r2": float(r2_score(y_true[mask], y_pred[mask])),
        }
    return out


@timed_step("train_pipeline")
def train_pipeline(
    split: DataSplits,
    feature_cols: List[str],
    runtime: Dict[str, object],
    reg_params: Dict[str, float],
    use_visibility_weighting: bool,
) -> Dict[str, object]:
    models: Dict[str, object] = {}

    X_train = split.train[feature_cols]
    X_val = split.val[feature_cols]

    validate_no_nan_inf(split.train, feature_cols, "Train features")
    validate_no_nan_inf(split.val, feature_cols, "Val features")

    for target_col in TARGET_COLUMNS:
        y_train = split.train[target_col]
        y_val = split.val[target_col]

        validate_no_nan_inf(split.train, [target_col], f"Train target {target_col}")
        validate_no_nan_inf(split.val, [target_col], f"Val target {target_col}")

        logger.info(f"Training model for {target_col}")
        sw = None
        if use_visibility_weighting and target_col == "visibility_target":
            sw, sw_stats = build_visibility_sample_weights(y_train)
            logger.info(
                "Visibility sample weights: low_vis=%d severe=%d min=%.1f max=%.1f mean=%.2f",
                sw_stats["low_visibility_count"],
                sw_stats["severe_visibility_count"],
                sw_stats["weight_min"],
                sw_stats["weight_max"],
                sw_stats["weight_mean"],
            )
        model = train_model(X_train, y_train, X_val, y_val, runtime, reg_params, sample_weight_train=sw)
        models[target_col] = model

        if hasattr(model, "feature_importances_"):
            imp = np.asarray(model.feature_importances_, dtype=np.float64)
            if imp.size == len(feature_cols):
                top_idx = np.argsort(imp)[::-1][:10]
                top_feats = [f"{feature_cols[i]}={imp[i]:.4f}" for i in top_idx]
                logger.info(f"Top features for {target_col}: {', '.join(top_feats)}")
                weather_feats = [c for c in feature_cols if "is_" in c or "weather" in c]
                if weather_feats:
                    weather_scores = sorted(
                        ((feature_cols[i], float(imp[i])) for i in range(len(feature_cols)) if feature_cols[i] in weather_feats),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:10]
                    logger.info(
                        "Weather-feature importance for %s: %s",
                        target_col,
                        ", ".join(f"{name}={score:.4f}" for name, score in weather_scores) if weather_scores else "none",
                    )

    return {
        "models": models,
        "feature_columns": feature_cols,
        "target_columns": TARGET_COLUMNS,
        "runtime": runtime,
    }


def mean_rmse(metrics: Dict[str, object]) -> float:
    vals = [metrics[t]["rmse"] for t in TARGET_COLUMNS]
    return float(np.mean(vals))


def focus_r2(metrics: Dict[str, object]) -> float:
    return float(
        np.mean(
            [
                metrics["wind_speed_target"]["r2"],
                metrics["visibility_target"]["r2"],
            ]
        )
    )


def metrics_to_rows(stage_name: str, metrics: Dict[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for t in TARGET_COLUMNS:
        rows.append(
            {
                "stage": stage_name,
                "target": t,
                "rmse": float(metrics[t]["rmse"]),
                "mae": float(metrics[t]["mae"]),
                "r2": float(metrics[t]["r2"]),
            }
        )
    return rows


def train_event_classifier(
    X_train: pd.DataFrame,
    y_train_event: pd.Series,
    X_val: pd.DataFrame,
    y_val_event: pd.Series,
    runtime: Dict[str, object],
):
    xgb = import_xgboost()

    y_arr = y_train_event.to_numpy(dtype=np.int32)
    pos = max(int(y_arr.sum()), 1)
    neg = max(len(y_arr) - pos, 1)
    scale_pos_weight = neg / pos

    clf = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=30,
        random_state=SEED,
        n_jobs=-1,
        tree_method=runtime["tree_method"],
        device=runtime["device"],
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
    )
    clf.fit(X_train, y_train_event.astype(int), eval_set=[(X_val, y_val_event.astype(int))], verbose=False)
    return clf


def attach_event_probability_features(
    split: DataSplits,
    runtime: Dict[str, object],
    feature_reference_df: pd.DataFrame,
    event_threshold: float = 1000.0,
) -> Tuple[DataSplits, object, List[str]]:
    y_train_event = (split.train["visibility_target"] < event_threshold).astype(int)
    y_val_event = (split.val["visibility_target"] < event_threshold).astype(int)

    base_feature_cols = get_feature_columns(split.train, missing_reference_df=feature_reference_df)
    event_clf = train_event_classifier(
        split.train[base_feature_cols],
        y_train_event,
        split.val[base_feature_cols],
        y_val_event,
        runtime,
    )

    updated_train = split.train.copy()
    updated_val = split.val.copy()
    updated_test = split.test.copy()

    for part in [updated_train, updated_val, updated_test]:
        part["low_visibility_event_prob"] = event_clf.predict_proba(part[base_feature_cols])[:, 1].astype("float32")
        part["low_vis_humidity"] = (part["low_visibility_event_prob"] * part["humidity"]).astype("float32")
        if "dew_proximity" in part.columns:
            part["low_vis_dew"] = (part["low_visibility_event_prob"] * part["dew_proximity"]).astype("float32")
        else:
            part["low_vis_dew"] = (part["low_visibility_event_prob"] * np.abs(part["temp"] - part["dew_point"])).astype("float32")

    updated_split = DataSplits(train=updated_train, val=updated_val, test=updated_test)
    return updated_split, event_clf, base_feature_cols


def create_severe_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 3000.0,
) -> Tuple[pd.DataFrame, pd.Series]:
    severe_mask = y.to_numpy(dtype=np.float64) < float(threshold)
    return X.loc[severe_mask].copy(), y.loc[severe_mask].copy()


def augment_severe_data(
    X: pd.DataFrame,
    y: pd.Series,
    repeats: int = 8,
    seed: int = SEED,
) -> Tuple[pd.DataFrame, pd.Series]:
    if X.empty:
        return X.copy(), y.copy()

    repeats = int(np.clip(repeats, 2, 10))
    rng = np.random.default_rng(seed)
    physical_bounds = {
        "temp": (-10.0, 55.0),
        "wind_speed": (0.0, 45.0),
        "humidity": (0.0, 100.0),
        "pressure": (950.0, 1050.0),
        "dew_point": (-20.0, 40.0),
        "visibility": (0.0, 12000.0),
    }
    noise_scales = {
        "temp": 0.8,
        "wind_speed": 1.0,
        "humidity": 2.0,
        "pressure": 0.6,
        "dew_point": 0.7,
        "visibility": 100.0,
    }

    frames = [X.copy()]
    targets = [y.copy()]

    perturbable_cols = [c for c in noise_scales if c in X.columns]
    for _ in range(repeats - 1):
        noisy = X.copy()
        for col in perturbable_cols:
            values = pd.to_numeric(noisy[col], errors="coerce").to_numpy(dtype=np.float64)
            values = values + rng.normal(0.0, noise_scales[col], size=len(values))
            lo, hi = physical_bounds[col]
            noisy[col] = np.clip(values, lo, hi).astype("float32")
        frames.append(noisy)
        targets.append(y.copy())

    augmented_X = pd.concat(frames, axis=0, ignore_index=True)
    augmented_y = pd.concat(targets, axis=0, ignore_index=True)
    return augmented_X, augmented_y


def evaluate_visibility_predictions(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    y_arr = y_true.to_numpy(dtype=np.float64)
    low_mask = y_arr < 5000.0
    severe_mask = y_arr < 1000.0

    def safe_metrics(mask: np.ndarray) -> Tuple[float, float, float]:
        if int(mask.sum()) == 0:
            return float("nan"), float("nan"), float("nan")
        return (
            float(r2_score(y_arr[mask], y_pred[mask])),
            float(mean_absolute_error(y_arr[mask], y_pred[mask])),
            float(np.sqrt(mean_squared_error(y_arr[mask], y_pred[mask]))),
        )

    overall_r2 = float(r2_score(y_arr, y_pred))
    overall_mae = float(mean_absolute_error(y_arr, y_pred))
    overall_rmse = float(np.sqrt(mean_squared_error(y_arr, y_pred)))
    low_r2, low_mae, low_rmse = safe_metrics(low_mask)
    severe_r2, severe_mae, severe_rmse = safe_metrics(severe_mask)

    return {
        "overall_r2": overall_r2,
        "overall_mae": overall_mae,
        "overall_rmse": overall_rmse,
        "low_visibility_r2": low_r2,
        "low_visibility_mae": low_mae,
        "low_visibility_rmse": low_rmse,
        "severe_visibility_r2": severe_r2,
        "severe_visibility_mae": severe_mae,
        "severe_visibility_rmse": severe_rmse,
        "low_visibility_count": int(low_mask.sum()),
        "severe_visibility_count": int(severe_mask.sum()),
    }


def compute_switch_diagnostics(y_true: pd.Series, severe_mask: np.ndarray) -> Dict[str, float]:
    y_arr = y_true.to_numpy(dtype=np.float64)
    actual_severe = y_arr < 1000.0
    actual_non_severe = ~actual_severe

    tp = int(np.sum(actual_severe & severe_mask))
    fp = int(np.sum(actual_non_severe & severe_mask))
    fn = int(np.sum(actual_severe & ~severe_mask))
    tn = int(np.sum(actual_non_severe & ~severe_mask))

    severe_total = int(actual_severe.sum())
    non_severe_total = int(actual_non_severe.sum())

    detection_rate = float(tp / severe_total) if severe_total > 0 else float("nan")
    false_positive_rate = float(fp / non_severe_total) if non_severe_total > 0 else float("nan")

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "actual_severe": severe_total,
        "actual_non_severe": non_severe_total,
        "severe_detection_rate": detection_rate,
        "false_positive_rate": false_positive_rate,
        "severe_model_usage_pct": float(severe_mask.mean() * 100.0),
    }


def predict_with_regime_switch(
    X: pd.DataFrame,
    general_model: object,
    severe_model: object,
    threshold: float = 0.1,
    event_prob_col: str = "low_visibility_event_prob",
    return_mask: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    if event_prob_col not in X.columns:
        raise ValueError(f"Missing required event probability column: {event_prob_col}")

    event_prob = pd.to_numeric(X[event_prob_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    vis_drop_rate = pd.to_numeric(X.get("vis_drop_rate", pd.Series(0.0, index=X.index)), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    severe_mask = (event_prob > float(threshold)) | (vis_drop_rate < -0.3)

    general_raw = np.asarray(general_model.predict(X), dtype=np.float64)
    severe_raw = np.asarray(severe_model.predict(X), dtype=np.float64)

    general_pred = clip_prediction("visibility_target", general_raw)
    severe_pred = clip_prediction("visibility_target", severe_raw)
    validate_prediction_stability("visibility_target", general_raw, general_pred)
    validate_prediction_stability("visibility_target", severe_raw, severe_pred)

    final_pred = np.where(severe_mask, severe_pred, general_pred)
    if return_mask:
        return final_pred, severe_mask
    return final_pred


def run_regime_model_pipeline(
    clean_df: pd.DataFrame,
    regime_threshold: float = 0.1,
    severe_train_threshold: float = 1500.0,
    severe_augment_repeats: int = 2,
    event_threshold: float = 1000.0,
    save_output_json: str = None,
) -> Dict[str, object]:
    set_seed(SEED)
    xgb = import_xgboost()
    runtime = get_gpu_runtime_params(xgb)
    logger.info(f"XGBoost GPU runtime: {runtime}")

    feat_df = add_features(_ensure_datetime_index(clean_df), add_enhanced_signals=True)
    feature_reference_df = feat_df.copy()
    all_df = add_targets(feat_df)
    split = split_chronological(all_df)
    split, event_clf, base_feature_cols = attach_event_probability_features(
        split,
        runtime,
        feature_reference_df,
        event_threshold=event_threshold,
    )

    feature_cols = get_feature_columns(split.train, missing_reference_df=feature_reference_df)
    logger.info("Regime feature count: %d", len(feature_cols))

    general_X_train = split.train[feature_cols]
    general_y_train = split.train["visibility_target"]
    general_X_val = split.val[feature_cols]
    general_y_val = split.val["visibility_target"]
    general_weights, general_weight_stats = build_visibility_sample_weights(general_y_train)
    logger.info(
        "General visibility weights: low_vis=%d severe=%d min=%.1f max=%.1f mean=%.2f",
        general_weight_stats["low_visibility_count"],
        general_weight_stats["severe_visibility_count"],
        general_weight_stats["weight_min"],
        general_weight_stats["weight_max"],
        general_weight_stats["weight_mean"],
    )
    general_model = train_model(
        general_X_train,
        general_y_train,
        general_X_val,
        general_y_val,
        runtime,
        {"gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0},
        sample_weight_train=general_weights,
    )

    X_train_severe, y_train_severe = create_severe_dataset(split.train[feature_cols], split.train["visibility_target"], threshold=severe_train_threshold)
    X_val_severe, y_val_severe = create_severe_dataset(split.val[feature_cols], split.val["visibility_target"], threshold=severe_train_threshold)
    if X_val_severe.empty:
        logger.warning("Severe validation subset is empty; falling back to full validation set for early stopping.")
        X_val_severe = general_X_val
        y_val_severe = general_y_val

    logger.info(
        "Severe sample count before augmentation: train=%d val=%d",
        len(X_train_severe),
        len(X_val_severe),
    )
    X_train_severe_aug, y_train_severe_aug = augment_severe_data(
        X_train_severe,
        y_train_severe,
        repeats=severe_augment_repeats,
        seed=SEED,
    )
    augmentation_factor = float(len(X_train_severe_aug) / max(len(X_train_severe), 1))
    logger.info(
        "Severe sample count after augmentation: train=%d (x%d, factor=%.2f)",
        len(X_train_severe_aug),
        severe_augment_repeats,
        augmentation_factor,
    )
    severe_weights, severe_weight_stats = build_visibility_sample_weights(y_train_severe_aug)
    logger.info(
        "Severe model weights: moderate_low=%d severe=%d min=%.1f max=%.1f mean=%.2f",
        severe_weight_stats["moderate_low_count"],
        severe_weight_stats["severe_visibility_count"],
        severe_weight_stats["weight_min"],
        severe_weight_stats["weight_max"],
        severe_weight_stats["weight_mean"],
    )
    severe_model = train_model(
        X_train_severe_aug,
        y_train_severe_aug,
        X_val_severe,
        y_val_severe,
        runtime,
        {"gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0},
        sample_weight_train=severe_weights,
    )

    X_test = split.test[feature_cols]
    event_prob_test = pd.to_numeric(X_test["low_visibility_event_prob"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    logger.info(
        "Event probability distribution on test: min=%.4f p25=%.4f p50=%.4f p75=%.4f max=%.4f",
        float(np.min(event_prob_test)),
        float(np.percentile(event_prob_test, 25)),
        float(np.percentile(event_prob_test, 50)),
        float(np.percentile(event_prob_test, 75)),
        float(np.max(event_prob_test)),
    )
    vis_drop_rate_test = pd.to_numeric(X_test["vis_drop_rate"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    logger.info(
        "vis_drop_rate distribution: min=%.4f p25=%.4f p50=%.4f p75=%.4f max=%.4f",
        float(np.min(vis_drop_rate_test)),
        float(np.percentile(vis_drop_rate_test, 25)),
        float(np.percentile(vis_drop_rate_test, 50)),
        float(np.percentile(vis_drop_rate_test, 75)),
        float(np.max(vis_drop_rate_test)),
    )
    logger.info("vis_drop_rate < -0.3 count: %d", int((vis_drop_rate_test < -0.3).sum()))
    logger.info(
        "Severe subset sizes: visibility<1000=%d | visibility<1500=%d",
        int((split.train["visibility_target"] < 1000.0).sum()),
        int((split.train["visibility_target"] < severe_train_threshold).sum()),
    )

    general_pred = clip_prediction("visibility_target", np.asarray(general_model.predict(X_test), dtype=np.float64))
    y_test = split.test["visibility_target"]
    general_metrics = evaluate_visibility_predictions(y_test, general_pred)

    threshold_grid = [0.05, 0.1, 0.2, 0.3]
    threshold_results: Dict[float, Dict[str, object]] = {}
    threshold_rows: List[Dict[str, object]] = []

    for threshold in threshold_grid:
        regime_pred, severe_mask = predict_with_regime_switch(
            X_test,
            general_model,
            severe_model,
            threshold=threshold,
            event_prob_col="low_visibility_event_prob",
            return_mask=True,
        )
        regime_metrics = evaluate_visibility_predictions(y_test, regime_pred)
        switch_diag = compute_switch_diagnostics(y_test, severe_mask)
        threshold_results[threshold] = {
            "metrics": regime_metrics,
            "diagnostics": switch_diag,
        }
        threshold_rows.append(
            {
                "threshold": threshold,
                **regime_metrics,
                **switch_diag,
            }
        )
        logger.info(
            "Threshold %.2f | severe_usage=%.2f%% | severe_detection=%.2f%% | false_positive=%.2f%% | severe_mae=%.3f | severe_r2=%.3f | overall_r2=%.3f",
            threshold,
            switch_diag["severe_model_usage_pct"],
            switch_diag["severe_detection_rate"] * 100.0,
            switch_diag["false_positive_rate"] * 100.0,
            regime_metrics["severe_visibility_mae"],
            regime_metrics["severe_visibility_r2"],
            regime_metrics["overall_r2"],
        )

    def format_threshold_table(rows: List[Dict[str, object]]) -> str:
        header = "| Threshold | Severe Use % | Severe Detect % | False Pos % | Overall R2 | Overall MAE | Low Vis R2 | Low Vis MAE | Severe R2 | Severe MAE |"
        sep = "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        body = []
        for row in rows:
            body.append(
                f"| {row['threshold']:.2f} | {row['severe_model_usage_pct']:.2f} | {row['severe_detection_rate'] * 100.0:.2f} | {row['false_positive_rate'] * 100.0:.2f} | {row['overall_r2']:.4f} | {row['overall_mae']:.4f} | {row['low_visibility_r2']:.4f} | {row['low_visibility_mae']:.4f} | {row['severe_visibility_r2']:.4f} | {row['severe_visibility_mae']:.4f} |"
            )
        return "\n".join([header, sep] + body)

    def select_best_threshold(results: Dict[float, Dict[str, object]]) -> Dict[str, float]:
        ranked = []
        for threshold, payload in results.items():
            metrics = payload["metrics"]
            ranked.append(
                {
                    "threshold": threshold,
                    "severe_mae": metrics["severe_visibility_mae"],
                    "severe_r2": metrics["severe_visibility_r2"],
                    "overall_r2": metrics["overall_r2"],
                    "overall_mae": metrics["overall_mae"],
                }
            )

        best_severe_mae = min(ranked, key=lambda x: x["severe_mae"])
        best_severe_r2 = max(ranked, key=lambda x: x["severe_r2"])
        acceptable = [r for r in ranked if np.isfinite(r["overall_r2"]) and r["overall_r2"] >= general_metrics["overall_r2"] - 0.02]
        if acceptable:
            best_balance = min(acceptable, key=lambda x: (x["severe_mae"], -x["severe_r2"], -x["overall_r2"]))
        else:
            best_balance = min(ranked, key=lambda x: (x["severe_mae"], -x["severe_r2"], -x["overall_r2"]))

        return {
            "best_for_severe_mae": float(best_severe_mae["threshold"]),
            "best_for_severe_r2": float(best_severe_r2["threshold"]),
            "best_balance": float(best_balance["threshold"]),
        }

    best_thresholds = select_best_threshold(threshold_results)
    recommended_threshold = best_thresholds["best_balance"]
    logger.info("Threshold comparison table:\n%s", format_threshold_table(threshold_rows))
    logger.info(
        "Best thresholds | severe_mae=%.2f | severe_r2=%.2f | balance=%.2f | recommended=%.2f",
        best_thresholds["best_for_severe_mae"],
        best_thresholds["best_for_severe_r2"],
        best_thresholds["best_balance"],
        recommended_threshold,
    )
    if threshold_rows:
        best_mae_row = min(threshold_rows, key=lambda r: r["severe_visibility_mae"])
        best_r2_row = max(threshold_rows, key=lambda r: r["severe_visibility_r2"])
        best_balance_row = min(
            threshold_rows,
            key=lambda r: (
                r["severe_visibility_mae"],
                -r["severe_visibility_r2"],
                -r["overall_r2"],
            ),
        )
        logger.info(
            "Recommendation summary: severe_MAE=%.2f (thr=%.2f), severe_R2=%.2f (thr=%.2f), balance=%.2f (thr=%.2f), production=%.2f",
            best_mae_row["severe_visibility_mae"],
            best_mae_row["threshold"],
            best_r2_row["severe_visibility_r2"],
            best_r2_row["threshold"],
            best_balance_row["overall_r2"],
            best_balance_row["threshold"],
            recommended_threshold,
        )

    comparison = {
        "general": general_metrics,
        "threshold_results": threshold_results,
        "best_thresholds": best_thresholds,
        "recommended_threshold": recommended_threshold,
        "threshold_rows": threshold_rows,
    }

    metrics = {
        "thresholds": {
            "regime_switch_default": float(regime_threshold),
            "severe_train": float(severe_train_threshold),
            "event_threshold": float(event_threshold),
            "evaluated": threshold_grid,
        },
        "counts": {
            "train_total": int(len(split.train)),
            "train_severe": int(len(X_train_severe)),
            "train_severe_augmented": int(len(X_train_severe_aug)),
            "test_total": int(len(split.test)),
        },
        "event_probability": {
            "min": float(np.min(event_prob_test)),
            "p25": float(np.percentile(event_prob_test, 25)),
            "median": float(np.percentile(event_prob_test, 50)),
            "p75": float(np.percentile(event_prob_test, 75)),
            "max": float(np.max(event_prob_test)),
        },
        "general": general_metrics,
        "regime": threshold_results,
        "comparison": comparison,
        "severe_model_usage_pct": {str(k): float(v["diagnostics"]["severe_model_usage_pct"]) for k, v in threshold_results.items()},
        "feature_count": int(len(feature_cols)),
        "recommendation": {
            "best_for_severe_mae": best_thresholds["best_for_severe_mae"],
            "best_for_severe_r2": best_thresholds["best_for_severe_r2"],
            "best_balance": best_thresholds["best_balance"],
            "production_threshold": recommended_threshold,
        },
    }

    recommended_pred, recommended_mask = predict_with_regime_switch(
        X_test,
        general_model,
        severe_model,
        threshold=recommended_threshold,
        event_prob_col="low_visibility_event_prob",
        return_mask=True,
    )
    recommended_diag = compute_switch_diagnostics(y_test, recommended_mask)

    result = {
        "predictions": {
            "general_visibility_target": general_pred.tolist(),
            "regime_visibility_target": recommended_pred.tolist(),
        },
        "metrics": metrics,
        "models": {
            "general": general_model,
            "severe": severe_model,
            "event_classifier": event_clf,
        },
        "feature_columns": feature_cols,
        "test_index": [pd.Timestamp(ts).isoformat() for ts in split.test.index],
        "recommended_threshold": recommended_threshold,
        "recommended_diagnostics": recommended_diag,
    }

    if save_output_json:
        out_path = Path(save_output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "predictions": result["predictions"],
                    "metrics": metrics,
                    "feature_count": int(len(feature_cols)),
                    "severe_model_usage_pct": metrics["severe_model_usage_pct"],
                    "recommended_threshold": recommended_threshold,
                    "threshold_results": threshold_results,
                    "recommended_diagnostics": recommended_diag,
                    "test_index": result["test_index"],
                },
                f,
                indent=2,
            )

    logger.info("General vs regime comparison: %s", json.dumps(comparison, indent=2))
    logger.info("Recommended threshold diagnostics: %s", json.dumps(recommended_diag, indent=2))
    return result


def run_stage(
    clean_df: pd.DataFrame,
    runtime: Dict[str, object],
    stage: StageConfig,
) -> Tuple[Dict[str, object], Dict[str, object], Dict[str, float], Dict[str, Dict[str, float]], pd.DataFrame]:
    logger.info(f"Running stage: {stage.name}")
    feat_df = add_features(
        clean_df,
        add_enhanced_signals=stage.add_enhanced_signals,
    )
    pre_target_df = feat_df.copy()
    all_df = add_targets(feat_df)
    split = split_chronological(all_df)

    if stage.add_event_probability_feature:
        split, event_clf, _ = attach_event_probability_features(
            split,
            runtime,
            pre_target_df,
            event_threshold=1000.0,
        )

    feature_cols = get_feature_columns(split.train, missing_reference_df=pre_target_df)

    bundle = train_pipeline(split, feature_cols, runtime, stage.reg_params, stage.use_visibility_weighting)
    X_test = split.test[feature_cols]
    preds = predict(bundle["models"], X_test)
    metrics = evaluate(split.test, preds)
    seg_metrics = evaluate_segments(
        split.test["visibility_target"].to_numpy(dtype=np.float64),
        preds["visibility_target"],
    )
    return bundle, metrics, metrics["visibility_summary"], seg_metrics, split.test


def format_comparison_table(rows: List[Dict[str, object]]) -> str:
    header = "| Stage | Target | RMSE | MAE | R2 |"
    sep = "|---|---|---:|---:|---:|"
    body = [
        f"| {r['stage']} | {r['target']} | {r['rmse']:.4f} | {r['mae']:.4f} | {r['r2']:.4f} |"
        for r in rows
    ]
    return "\n".join([header, sep] + body)


def format_low_visibility_table(rows: List[Dict[str, object]]) -> str:
    header = "| Stage | Overall R2 | Low Vis R2 | Severe Vis R2 | Visibility MAE |"
    sep = "|---|---:|---:|---:|---:|"
    body = [
        f"| {r['stage']} | {r['overall_r2']:.4f} | {r['low_visibility_r2']:.4f} | {r['severe_visibility_r2']:.4f} | {r['visibility_mae']:.4f} |"
        for r in rows
    ]
    return "\n".join([header, sep] + body)


def format_segment_table(rows: List[Dict[str, object]]) -> str:
    header = "| Stage | Segment | Count | RMSE | MAE | R2 |"
    sep = "|---|---|---:|---:|---:|---:|"
    body: List[str] = []
    for r in rows:
        body.append(
            f"| {r['stage']} | {r['segment']} | {r['count']} | {r['rmse']:.4f} | {r['mae']:.4f} | {r['r2']:.4f} |"
        )
    return "\n".join([header, sep] + body)


def save_forecast_json(index: pd.DatetimeIndex, preds: Dict[str, np.ndarray], output_json: str) -> None:
    out = []
    for i, ts in enumerate(index):
        out.append(
            {
                "timestamp": pd.Timestamp(ts).isoformat(),
                "forecast_timestamp": (pd.Timestamp(ts) + pd.Timedelta(hours=6)).isoformat(),
                "temp": float(preds["temp_target"][i]),
                "wind_speed": float(preds["wind_speed_target"][i]),
                "visibility": float(preds["visibility_target"][i]),
                "pressure": float(preds["pressure_target"][i]),
                "humidity": float(preds["humidity_target"][i]),
            }
        )

    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


@timed_step("save_plots")
def save_plots(
    index: pd.DatetimeIndex,
    y_df: pd.DataFrame,
    preds: Dict[str, np.ndarray],
    plots_dir: str,
    metrics: Dict[str, object],
    interactive: bool = False,
) -> List[str]:
    out_dir = Path(plots_dir)
    if not interactive:
        out_dir.mkdir(parents=True, exist_ok=True)

    saved_files: List[str] = []

    for base_name in CORE_TARGETS:
        target_col = f"{base_name}_target"
        actual = y_df[target_col].to_numpy(dtype=np.float64)
        predicted = preds[target_col]
        residuals = actual - predicted

        fig_ts, ax_ts = plt.subplots(figsize=(14, 4))
        ax_ts.plot(index, actual, label="Actual", linewidth=1.2)
        ax_ts.plot(index, predicted, label="Predicted", linewidth=1.2, alpha=0.85)
        ax_ts.set_title(f"{base_name} - Actual vs Predicted")
        ax_ts.set_xlabel("Datetime")
        ax_ts.set_ylabel(base_name)
        ax_ts.grid(True, alpha=0.3)
        ax_ts.legend()
        fig_ts.tight_layout()
        if interactive:
            fig_ts.show()
        else:
            ts_path = out_dir / f"{base_name}_timeseries.png"
            fig_ts.savefig(ts_path, dpi=150)
            saved_files.append(str(ts_path))
            plt.close(fig_ts)

        fig_res, (ax_line, ax_hist) = plt.subplots(2, 1, figsize=(14, 6), sharex=False)
        ax_line.plot(index, residuals, color="tab:orange", linewidth=1.0)
        ax_line.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
        ax_line.set_title(f"{base_name} - Residuals (Actual - Predicted)")
        ax_line.set_xlabel("Datetime")
        ax_line.set_ylabel("Residual")
        ax_line.grid(True, alpha=0.3)

        ax_hist.hist(residuals, bins=60, color="tab:blue", alpha=0.75)
        ax_hist.set_title(f"{base_name} - Residual Histogram")
        ax_hist.set_xlabel("Residual")
        ax_hist.set_ylabel("Count")
        ax_hist.grid(True, alpha=0.3)

        fig_res.tight_layout()
        if interactive:
            fig_res.show()
        else:
            res_path = out_dir / f"{base_name}_residuals.png"
            fig_res.savefig(res_path, dpi=150)
            saved_files.append(str(res_path))
            plt.close(fig_res)

    fig_dash, axes = plt.subplots(len(CORE_TARGETS), 1, figsize=(16, 16), sharex=True)
    for ax, base_name in zip(axes, CORE_TARGETS):
        target_col = f"{base_name}_target"
        actual = y_df[target_col].to_numpy(dtype=np.float64)
        predicted = preds[target_col]
        m = metrics.get(target_col, {})
        rmse = m.get("rmse", float("nan"))
        mae = m.get("mae", float("nan"))
        r2 = m.get("r2", float("nan"))
        ax.plot(index, actual, label="Actual", linewidth=1.0)
        ax.plot(index, predicted, label="Predicted", linewidth=1.0, alpha=0.85)
        ax.set_title(f"{base_name}: Actual vs Predicted | RMSE={rmse:.3f}, MAE={mae:.3f}, R2={r2:.3f}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Datetime")

    fig_dash.tight_layout()
    if interactive:
        fig_dash.show()
        plt.show()
    else:
        dash_path = out_dir / "combined_dashboard.png"
        fig_dash.savefig(dash_path, dpi=150)
        plt.close(fig_dash)
        saved_files.append(str(dash_path))

    return saved_files


@timed_step("run_pipeline")
def run_pipeline(
    input_csv: str,
    model_file: str,
    eval_output_json: str,
    forecast_output_json: str,
    interactive_plots: bool = False,
) -> None:
    set_seed(SEED)

    xgb = import_xgboost()
    runtime = get_gpu_runtime_params(xgb)
    logger.info(f"XGBoost GPU runtime: {runtime}")

    clean_df = load_and_clean(input_csv)
    logger.info("Visibility class distribution: %s", visibility_distribution(clean_df))
    validate_no_nan_inf(clean_df, [c for c in BASE_CONTINUOUS + BASE_BINARY if c in clean_df.columns], "Clean input")

    wind_percentile_threshold = calibrate_wind_speed_threshold(clean_df)
    clean_df = apply_wind_speed_limits(clean_df, wind_percentile_threshold)
    logger.info("Applied wind speed hard cap and train-based percentile clipping before feature engineering.")

    stages = [
        StageConfig(
            name="Baseline",
            add_enhanced_signals=False,
            add_event_probability_feature=False,
            use_visibility_weighting=False,
            reg_params={"gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0},
        ),
        StageConfig(
            name="+PhysicsSignals",
            add_enhanced_signals=True,
            add_event_probability_feature=False,
            use_visibility_weighting=False,
            reg_params={"gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0},
        ),
        StageConfig(
            name="+EventProbAndWeightedVis",
            add_enhanced_signals=True,
            add_event_probability_feature=True,
            use_visibility_weighting=True,
            reg_params={"gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0},
        ),
    ]

    comparison_rows: List[Dict[str, object]] = []
    low_visibility_rows: List[Dict[str, object]] = []
    segment_rows: List[Dict[str, object]] = []
    stage_metrics_by_name: Dict[str, Dict[str, object]] = {}
    accepted_stage_name = ""
    accepted_bundle: Dict[str, object] = {}
    accepted_test_df = pd.DataFrame()
    accepted_metrics: Dict[str, object] = {}
    best_rank = None

    for stage in stages:
        try:
            bundle, stage_metrics, stage_low_vis, stage_seg, stage_test_df = run_stage(clean_df, runtime, stage)
        except Exception as exc:
            logger.warning(f"Discarding stage {stage.name} due to stability/runtime issue: {exc}")
            continue

        comparison_rows.extend(metrics_to_rows(stage.name, stage_metrics))
        low_visibility_rows.append({"stage": stage.name, **stage_low_vis})
        stage_metrics_by_name[stage.name] = stage_metrics
        for seg_name, seg_vals in stage_seg.items():
            segment_rows.append({"stage": stage.name, "segment": seg_name, **seg_vals})
        vis_summary = stage_metrics["visibility_summary"]
        logger.info(
            "Stage %s visibility summary: overall_r2=%.4f low_vis_r2=%.4f severe_r2=%.4f mae=%.4f",
            stage.name,
            vis_summary["overall_r2"],
            vis_summary["low_visibility_r2"],
            vis_summary["severe_visibility_r2"],
            vis_summary["visibility_mae"],
        )

        stage_rank = (
            vis_summary["low_visibility_r2"],
            vis_summary["severe_visibility_r2"],
            vis_summary["overall_r2"],
            -vis_summary["visibility_mae"],
        )
        logger.info(f"Stage {stage.name} selection rank: {stage_rank}")

        if best_rank is None or stage_rank > best_rank:
            best_rank = stage_rank
            accepted_stage_name = stage.name
            accepted_bundle = bundle
            accepted_test_df = stage_test_df
            accepted_metrics = stage_metrics

    if not accepted_bundle:
        raise RuntimeError("No stable stage was accepted.")

    accepted_feature_cols = accepted_bundle["feature_columns"]
    accepted_preds = predict(accepted_bundle["models"], accepted_test_df[accepted_feature_cols])

    eval_metrics = {
        "final_stage": accepted_stage_name,
        "final_metrics": accepted_metrics,
        "comparison_rows": comparison_rows,
        "comparison_table": format_comparison_table(comparison_rows),
        "low_visibility_rows": low_visibility_rows,
        "low_visibility_table": format_low_visibility_table(low_visibility_rows),
        "segment_rows": segment_rows,
        "segment_table": format_segment_table(segment_rows),
        "before_after_comparison": {},
    }

    baseline_metrics = stage_metrics_by_name.get("Baseline")
    if baseline_metrics is not None:
        baseline_vis = baseline_metrics["visibility_summary"]
        final_vis = accepted_metrics["visibility_summary"]
        eval_metrics["before_after_comparison"] = {
            "baseline": baseline_vis,
            "final": final_vis,
            "delta": {
                "overall_r2": final_vis["overall_r2"] - baseline_vis["overall_r2"],
                "low_visibility_r2": final_vis["low_visibility_r2"] - baseline_vis["low_visibility_r2"],
                "severe_visibility_r2": final_vis["severe_visibility_r2"] - baseline_vis["severe_visibility_r2"],
                "visibility_mae": final_vis["visibility_mae"] - baseline_vis["visibility_mae"],
            },
        }

    model_bundle_path = Path(model_file)
    model_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(accepted_bundle, model_bundle_path)

    eval_path = Path(eval_output_json)
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with eval_path.open("w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2)

    save_forecast_json(accepted_test_df.index, accepted_preds, forecast_output_json)
    plot_files = save_plots(
        accepted_test_df.index,
        accepted_test_df,
        accepted_preds,
        plots_dir="artifacts/plots",
        metrics=accepted_metrics,
        interactive=interactive_plots,
    )

    summary = {
        "rows": {
            "clean": int(len(clean_df)),
            "features": int(len(accepted_feature_cols)),
            "test": int(len(accepted_test_df)),
        },
        "num_models": int(len(accepted_bundle["models"])),
        "final_stage": accepted_stage_name,
        "model_file": str(model_bundle_path),
        "eval_output": str(eval_path),
        "forecast_output": str(Path(forecast_output_json)),
        "plots_dir": None if interactive_plots else "artifacts/plots",
        "plot_files": plot_files,
        "interactive_plots": bool(interactive_plots),
    }
    logger.info("Comparison table:")
    logger.info(eval_metrics["comparison_table"])
    logger.info("Low-visibility table:")
    logger.info(eval_metrics["low_visibility_table"])
    logger.info("Segment table:")
    logger.info(eval_metrics["segment_table"])
    if eval_metrics["before_after_comparison"]:
        logger.info("Before vs after visibility comparison:")
        logger.info(json.dumps(eval_metrics["before_after_comparison"], indent=2))
    logger.info("Run summary:")
    logger.info(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stable and efficient CSMI weather forecasting pipeline.")
    parser.add_argument("--input", default="clean_weather_data.csv", help="Input CSV path with weather observations.")
    parser.add_argument(
        "--run-regime-model",
        action="store_true",
        help="Run the two-model regime-based visibility pipeline instead of the standard multi-target pipeline.",
    )
    parser.add_argument(
        "--regime-threshold",
        type=float,
        default=0.1,
        help="Event-probability threshold used to switch to the severe model.",
    )
    parser.add_argument(
        "--regime-severe-threshold",
        type=float,
        default=3000.0,
        help="Visibility threshold used to define the severe training subset.",
    )
    parser.add_argument(
        "--regime-augment-repeats",
        type=int,
        default=6,
        help="How many times to repeat severe rows during augmentation.",
    )
    parser.add_argument(
        "--regime-output-json",
        default="artifacts/advanced/regime_visibility_results.json",
        help="Optional JSON output path for regime pipeline predictions and metrics.",
    )
    parser.add_argument(
        "--model-file",
        default="artifacts/advanced/stable_models.joblib",
        help="Single artifact file containing trained models and metadata.",
    )
    parser.add_argument(
        "--eval-output",
        default="artifacts/advanced/eval_metrics_stable.json",
        help="Evaluation metrics output JSON path.",
    )
    parser.add_argument(
        "--forecast-output",
        default="artifacts/advanced/forecast_stable.json",
        help="Forecast output JSON path.",
    )
    parser.add_argument(
        "--interactive-plots",
        action="store_true",
        help="Open interactive matplotlib plot windows instead of saving PNG files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_regime_model:
        clean_df = load_and_clean(args.input)
        run_regime_model_pipeline(
            clean_df,
            regime_threshold=args.regime_threshold,
            severe_train_threshold=args.regime_severe_threshold,
            severe_augment_repeats=args.regime_augment_repeats,
            save_output_json=args.regime_output_json,
        )
    else:
        run_pipeline(
            input_csv=args.input,
            model_file=args.model_file,
            eval_output_json=args.eval_output,
            forecast_output_json=args.forecast_output,
            interactive_plots=args.interactive_plots,
        )


if __name__ == "__main__":
    main()