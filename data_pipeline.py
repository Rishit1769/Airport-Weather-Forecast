import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FREQ = "30min"
TRAIN_END = "2024-01-01"
LAG_STEPS = [1, 2, 3, 6, 12]
ROLLING_MEAN_WINDOWS = [3, 6]
ROLLING_STD_WINDOWS = [3, 6, 12]

BASE_CONTINUOUS = [
    "wind_dir",
    "wind_speed",
    "wind_gust",
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
    "wind_gust": (0.0, 80.0),
    "visibility": (0.0, 12000.0),
    "temp": (-10.0, 55.0),
    "dew_point": (-20.0, 40.0),
    "humidity": (0.0, 100.0),
    "pressure": (950.0, 1050.0),
    "cloud_cover": (0.0, 8.0),
}


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime"]).set_index("datetime")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Expected a DatetimeIndex or a datetime column.")
    return df.sort_index()


def _encode_weather_codes(df: pd.DataFrame) -> pd.DataFrame:
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

    for base_col, code_col in [("is_haze", "is_haze_code"), ("is_rain", "is_rain_code")]:
        base = pd.to_numeric(df.get(base_col, 0), errors="coerce").fillna(0).astype("int8")
        df[base_col] = np.maximum(base.to_numpy(), df[code_col].to_numpy()).astype("int8")
    return df


def load_and_clean(input_csv: str) -> pd.DataFrame:
    raw_df = pd.read_csv(input_csv)
    if "dew_point" not in raw_df.columns:
        if "td" in raw_df.columns:
            raw_df["dew_point"] = raw_df["td"]
        else:
            temp = pd.to_numeric(raw_df["temp"], errors="coerce")
            humidity = pd.to_numeric(raw_df["humidity"], errors="coerce").clip(0.1, 100.0)
            b, c = 17.62, 243.12
            gamma = (b * temp / (c + temp)) + np.log(humidity / 100.0)
            raw_df["dew_point"] = (c * gamma) / (b - gamma)

    raw_gust_coverage = (
        float(pd.to_numeric(raw_df["wind_gust"], errors="coerce").notna().mean())
        if "wind_gust" in raw_df.columns
        else 0.0
    )
    raw_rows = len(raw_df)
    df = _ensure_datetime_index(raw_df)
    full_index = pd.date_range(df.index.min(), df.index.max(), freq=FREQ)
    df = df.reindex(full_index)

    if "pressure" in df.columns:
        df["pressure"] = pd.to_numeric(df["pressure"], errors="coerce")
        df.loc[df["pressure"].diff().abs() > 10.0, "pressure"] = np.nan
    if "wind_speed" in df.columns:
        df["wind_speed"] = pd.to_numeric(df["wind_speed"], errors="coerce")
        df.loc[df["wind_speed"].diff().abs() > 30.0, "wind_speed"] = np.nan
        df["wind_speed"] = df["wind_speed"].clip(upper=45.0)

    for col in BASE_CONTINUOUS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].clip(*PHYSICAL_BOUNDS[col])
            df[col] = df[col].interpolate(
                method="time",
                limit=2,
                limit_direction="both",
                limit_area="inside",
            )
    for col in BASE_BINARY:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].ffill(limit=2).bfill(limit=2).clip(0, 1)

    df = _encode_weather_codes(df.ffill(limit=4).bfill(limit=2))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    observed_gust = (
        pd.to_numeric(df["wind_gust"], errors="coerce").clip(0.0, 80.0)
        if "wind_gust" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    proxy_gust = (df["wind_speed"] * 1.4).clip(0.0, 80.0)
    if raw_gust_coverage > 0.30:
        df["wind_gust"] = observed_gust.combine_first(proxy_gust)
    else:
        df["wind_gust"] = proxy_gust
        logger.warning("Using wind_gust proxy because observed coverage is %.2f%%.", raw_gust_coverage * 100)

    train_wind = df.loc[(df.index >= "2016-01-01") & (df.index < TRAIN_END), "wind_speed"]
    threshold_source = train_wind if not train_wind.empty else df["wind_speed"]
    wind_threshold = min(45.0, float(np.nanpercentile(threshold_source, 99.9)))
    df["wind_speed"] = df["wind_speed"].clip(upper=wind_threshold)

    df["temp_dew_diff"] = (df["temp"] - df["dew_point"]).clip(-30.0, 60.0)
    df["pressure_change"] = df["pressure"].diff().clip(-20.0, 20.0).fillna(0.0)
    df["wind_speed_change"] = df["wind_speed"].diff().clip(-40.0, 40.0).fillna(0.0)
    logger.info("Loaded %d rows; retained %d half-hour observations.", raw_rows, len(df))
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_datetime_index(df).copy()
    df["wind_dir_sin"] = np.sin(np.radians(df["wind_dir"]))
    df["wind_dir_cos"] = np.cos(np.radians(df["wind_dir"]))
    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12.0)

    stable_cols = [
        "temp",
        "wind_speed",
        "wind_gust",
        "visibility",
        "pressure",
        "humidity",
        "wind_dir_sin",
        "wind_dir_cos",
    ]
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

    df["humidity_wind"] = (df["humidity"] * df["wind_speed"]).clip(0.0, 8000.0)
    df["pressure_humidity"] = (df["pressure"] * df["humidity"]).clip(0.0, 120000.0)
    df["temp_humidity"] = (df["temp"] * df["humidity"]).clip(-1000.0, 6000.0)
    df["humidity_temperature"] = df["temp_humidity"]
    df["dew_point_depression"] = df["temp"] - df["dew_point"]

    humidity_for_wet_bulb = df["humidity"].clip(5.0, 99.0)
    df["wet_bulb"] = (
        df["temp"] * np.arctan(0.151977 * np.sqrt(humidity_for_wet_bulb + 8.313659))
        + np.arctan(df["temp"] + humidity_for_wet_bulb)
        - np.arctan(humidity_for_wet_bulb - 1.676331)
        + 0.00391838 * humidity_for_wet_bulb**1.5 * np.arctan(0.023101 * humidity_for_wet_bulb)
        - 4.686035
    )
    df["pressure_tendency_3h"] = df["pressure"].diff(6)
    df["pressure_tendency_sign"] = np.sign(df["pressure_tendency_3h"])
    visibility_trend = df["visibility"] - df["visibility"].shift(1)
    df["visibility_trend"] = visibility_trend.clip(-5000.0, 5000.0)
    df["visibility_acceleration"] = (visibility_trend - visibility_trend.shift(1)).clip(-5000.0, 5000.0)

    new_cols: Dict[str, pd.Series] = {}
    for col in stable_cols:
        for step in LAG_STEPS:
            new_cols[f"{col}_lag_{step}"] = df[col].shift(step)
        for window in ROLLING_MEAN_WINDOWS:
            new_cols[f"{col}_rolling_mean_{window}"] = df[col].rolling(window).mean()
        for window in ROLLING_STD_WINDOWS:
            new_cols[f"{col}_rolling_std_{window}"] = df[col].rolling(window).std()

    for step in [1, 2, 3, 6]:
        new_cols[f"dew_point_depression_lag_{step}"] = df["dew_point_depression"].shift(step)
    for step in [1, 2]:
        new_cols[f"wet_bulb_lag_{step}"] = df["wet_bulb"].shift(step)

    vis_lag_1 = df["visibility"].shift(1)
    vis_lag_3 = df["visibility"].shift(3)
    vis_lag_6 = df["visibility"].shift(6)
    for step, lag in [(1, vis_lag_1), (3, vis_lag_3), (6, vis_lag_6)]:
        new_cols[f"visibility_lag_{step}"] = lag
        new_cols[f"wind_speed_x_visibility_lag_{step}"] = df["wind_speed"] * lag
    for window in [3, 6, 12]:
        new_cols[f"visibility_rolling_mean_{window}"] = df["visibility"].rolling(window).mean()
        new_cols[f"visibility_rolling_std_{window}"] = df["visibility"].rolling(window).std()

    new_cols["vis_drop_1"] = df["visibility"] - vis_lag_1
    new_cols["vis_drop_3"] = df["visibility"] - vis_lag_3
    new_cols["vis_drop_rate"] = new_cols["vis_drop_1"] / (vis_lag_1 + 1.0)
    new_cols["high_humidity_flag"] = (df["humidity"] > 90.0).astype("float32")
    new_cols["humidity_spike"] = df["humidity"] - df["humidity"].shift(3)
    new_cols["low_wind_flag"] = (df["wind_speed"] < 2.0).astype("float32")
    dew_gap = np.abs(df["temp"] - df["dew_point"])
    new_cols["dew_gap"] = dew_gap
    new_cols["dew_gap_lag_3"] = dew_gap.shift(3)
    new_cols["dew_gap_change"] = dew_gap - dew_gap.shift(3)
    new_cols["pressure_drop_fast"] = df["pressure"] - df["pressure"].shift(3)
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    hour = df.index.hour
    dew_proximity = np.abs(df["temp"] - df["dew_point"])
    enhanced_cols = {
        "dew_proximity": dew_proximity.clip(0.0, 30.0),
        "near_dew_flag": (dew_proximity < 2.0).astype("float32"),
        "vis_regime": pd.cut(
            df["visibility"],
            bins=[0.0, 1000.0, 3000.0, 8000.0, 12000.0],
            labels=[0, 1, 2, 3],
            include_lowest=True,
            right=False,
        ).astype("float32"),
        "pressure_change_3h": (df["pressure"] - df["pressure"].shift(6)).clip(-20.0, 20.0),
        "morning_humidity": (df["humidity"] * ((hour >= 4) & (hour <= 8))).clip(0.0, 100.0),
    }
    low_vis = (df["visibility"] < 3000.0).astype("int32")
    enhanced_cols["low_visibility_flag"] = low_vis.astype("float32")
    enhanced_cols["low_visibility_streak"] = low_vis.groupby((low_vis == 0).cumsum()).cumsum().clip(0.0, 96.0)
    df = pd.concat([df, pd.DataFrame(enhanced_cols, index=df.index)], axis=1)

    df = df[df["pressure_change"].abs() < 20.0]
    df = df[df["wind_speed_change"].abs() < 40.0]
    df = df.dropna().copy()
    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype("float32")
    return df


def get_engineered_data(input_csv: str = "data/clean_weather_data.csv") -> pd.DataFrame:
    """Load, clean, and engineer the shared weather frame exactly once."""
    logger.info("Building shared engineered weather frame.")
    return add_features(load_and_clean(input_csv))
