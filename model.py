import time
import random
import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None


CONFIG = {
    "CSV_FILE": "clean_weather_data.csv",
    "SEED": 42,
    "SEQ_LENGTH": 72,
    "PRED_STEP": 12,
    "TRAIN_RATIO": 0.70,
    "VAL_RATIO": 0.15,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def build_features(csv_path):
    df = pd.read_csv(csv_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    if "humidity" not in df.columns:
        df["humidity"] = 60.0

    vapor = (df["humidity"] / 100.0) * 6.105 * np.exp((17.27 * df["temp"]) / (237.7 + df["temp"]))
    df["apparent_temp"] = df["temp"] + 0.33 * vapor - 4.0

    hr = df["datetime"].dt.hour
    dow = df["datetime"].dt.dayofweek
    doy = df["datetime"].dt.dayofyear

    hr_ang = 2.0 * np.pi * hr / 24.0
    doy_ang = 2.0 * np.pi * doy / 365.25

    df["hr_sin"] = np.sin(hr_ang)
    df["hr_cos"] = np.cos(hr_ang)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    df["doy_sin"] = np.sin(doy_ang)
    df["doy_cos"] = np.cos(doy_ang)

    # Solar zenith proxy from harmonic hour/year components.
    solar_decl = 0.409 * np.sin(doy_ang - 1.39)
    cos_zen = np.cos(hr_ang) * np.cos(solar_decl)
    cos_zen = np.clip(cos_zen, -1.0, 1.0)
    df["solar_zenith_proxy"] = np.arccos(cos_zen) / np.pi

    df["temp_diff_1"] = df["temp"].diff().fillna(0)
    df["pressure_diff_1"] = df["pressure"].diff().fillna(0)
    df["wind_diff_1"] = df["wind_speed"].diff().fillna(0)

    # Discrete Laplacian captures acceleration/inflection in heating-cooling cycles.
    df["temp_laplacian"] = (df["temp"].shift(-1) - 2.0 * df["temp"] + df["temp"].shift(1)).fillna(0)

    for w in [6, 12, 24, 48]:
        df[f"temp_mean_{w}"] = df["temp"].rolling(w).mean().bfill()
        df[f"temp_std_{w}"] = df["temp"].rolling(w).std().bfill()
        df[f"temp_min_{w}"] = df["temp"].rolling(w).min().bfill()
        df[f"temp_max_{w}"] = df["temp"].rolling(w).max().bfill()

    for lag in [1, 2, 6, 12]:
        df[f"temp_lag_{lag}"] = df["temp"].shift(lag).bfill()

    feature_cols = [
        "temp",
        "apparent_temp",
        "pressure",
        "wind_speed",
        "humidity",
        "hr_sin",
        "hr_cos",
        "dow_sin",
        "dow_cos",
        "doy_sin",
        "doy_cos",
        "solar_zenith_proxy",
        "temp_diff_1",
        "pressure_diff_1",
        "wind_diff_1",
        "temp_laplacian",
    ]

    feature_cols += [c for c in df.columns if c.startswith("temp_mean_")]
    feature_cols += [c for c in df.columns if c.startswith("temp_std_")]
    feature_cols += [c for c in df.columns if c.startswith("temp_min_")]
    feature_cols += [c for c in df.columns if c.startswith("temp_max_")]
    feature_cols += [c for c in df.columns if c.startswith("temp_lag_")]

    return df, feature_cols


def create_sequences(data_scaled, seq_len, pred_step):
    x_seq, y_seq, target_idx = [], [], []
    max_start = len(data_scaled) - seq_len - pred_step + 1
    for i in range(max_start):
        t_idx = i + seq_len + pred_step - 1
        x_seq.append(data_scaled[i : i + seq_len])
        y_seq.append(data_scaled[t_idx, 0])
        target_idx.append(t_idx)
    return (
        np.asarray(x_seq, dtype=np.float32),
        np.asarray(y_seq, dtype=np.float32),
        np.asarray(target_idx, dtype=np.int64),
    )


def flatten_sequences(x_seq, feature_cols, seq_len):
    x_flat = x_seq.reshape(x_seq.shape[0], -1)
    flat_cols = []
    for t in range(seq_len):
        lag = seq_len - 1 - t
        for feat in feature_cols:
            flat_cols.append(f"{feat}_t-{lag}")
    return x_flat, flat_cols


def inverse_temp_scale(scaler, feature_count, vec):
    dummy = np.zeros((len(vec), feature_count), dtype=np.float32)
    dummy[:, 0] = vec
    return scaler.inverse_transform(dummy)[:, 0]


def smape(y_true, y_pred):
    num = 2.0 * np.abs(y_true - y_pred)
    den = np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-3)
    return float(np.mean(num / den) * 100.0)


def evaluate_model(name, model, x_train, y_train, x_test, y_test, scaler, feature_count):
    start = time.time()
    model.fit(x_train, y_train)
    preds_scaled = model.predict(x_test)
    train_seconds = time.time() - start

    y_pred = inverse_temp_scale(scaler, feature_count, preds_scaled)
    y_true = inverse_temp_scale(scaler, feature_count, y_test)

    metrics = {
        "model": name,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "smape": smape(y_true, y_pred),
        "train_seconds": train_seconds,
    }

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=np.float64)
    else:
        importances = np.zeros(x_train.shape[1], dtype=np.float64)

    return metrics, y_true, y_pred, importances


def build_models(seed):
    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            n_jobs=-1,
            random_state=seed,
        )
    }

    if XGBRegressor is not None:
        models["XGBoost"] = XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            tree_method="gpu_hist",
            random_state=seed,
            objective="reg:squarederror",
            eval_metric="rmse",
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.9,
        )

    if LGBMRegressor is not None:
        models["LightGBM"] = LGBMRegressor(
            device="gpu",
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=64,
            objective="regression",
            random_state=seed,
            subsample=0.9,
            colsample_bytree=0.9,
        )

    return models


def fit_with_gpu_fallback(name, model, x_train, y_train):
    try:
        model.fit(x_train, y_train)
        return model, False
    except Exception as ex:
        msg = str(ex).lower()
        gpu_issue = "gpu" in msg or "cuda" in msg or "opencl" in msg
        if not gpu_issue:
            raise

        if name == "XGBoost":
            fallback = XGBRegressor(
                n_estimators=1000,
                learning_rate=0.05,
                tree_method="hist",
                random_state=CONFIG["SEED"],
                objective="reg:squarederror",
                eval_metric="rmse",
                max_depth=8,
                subsample=0.9,
                colsample_bytree=0.9,
            )
            fallback.fit(x_train, y_train)
            return fallback, True

        if name == "LightGBM":
            fallback = LGBMRegressor(
                device="cpu",
                n_estimators=1000,
                learning_rate=0.03,
                num_leaves=64,
                objective="regression",
                random_state=CONFIG["SEED"],
                subsample=0.9,
                colsample_bytree=0.9,
            )
            fallback.fit(x_train, y_train)
            return fallback, True

        raise


if __name__ == "__main__":
    set_seed(CONFIG["SEED"])

    df, feature_cols = build_features(CONFIG["CSV_FILE"])
    raw = df[feature_cols].values

    n = len(raw)
    train_cut = int(n * CONFIG["TRAIN_RATIO"])
    val_cut = int(n * (CONFIG["TRAIN_RATIO"] + CONFIG["VAL_RATIO"]))

    scaler = RobustScaler()
    scaler.fit(raw[:train_cut])
    data_scaled = scaler.transform(raw)

    x_seq, y, target_idx = create_sequences(data_scaled, CONFIG["SEQ_LENGTH"], CONFIG["PRED_STEP"])
    x_flat, flat_cols = flatten_sequences(x_seq, feature_cols, CONFIG["SEQ_LENGTH"])

    train_mask = target_idx < train_cut
    val_mask = (target_idx >= train_cut) & (target_idx < val_cut)
    test_mask = target_idx >= val_cut

    x_train = x_flat[train_mask]
    y_train = y[train_mask]
    x_val = x_flat[val_mask]
    y_val = y[val_mask]
    x_test = x_flat[test_mask]
    y_test = y[test_mask]

    # Train on train+val for final benchmark, test remains untouched.
    x_train_full = np.concatenate([x_train, x_val], axis=0)
    y_train_full = np.concatenate([y_train, y_val], axis=0)

    test_target_idx = target_idx[test_mask]
    humidity_test = df.loc[test_target_idx, "humidity"].values
    test_hours = df.loc[test_target_idx, "datetime"].dt.hour.values

    models = build_models(CONFIG["SEED"])
    if len(models) < 3:
        missing = []
        if XGBRegressor is None:
            missing.append("xgboost")
        if LGBMRegressor is None:
            missing.append("lightgbm")
        raise ImportError(
            "Missing required libraries for full benchmark: " + ", ".join(missing)
        )

    results = []
    predictions = {}
    importances = {}

    print(
        f"Benchmarking tree models | train={len(x_train_full)} test={len(x_test)} | "
        f"features={x_train_full.shape[1]}"
    )

    for name, model in models.items():
        start_fit = time.time()
        model_fitted, used_fallback = fit_with_gpu_fallback(name, model, x_train_full, y_train_full)
        fit_seconds = time.time() - start_fit

        preds_scaled = model_fitted.predict(x_test)
        y_pred = inverse_temp_scale(scaler, len(feature_cols), preds_scaled)
        y_true = inverse_temp_scale(scaler, len(feature_cols), y_test)

        metrics = {
            "model": name,
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)),
            "smape": smape(y_true, y_pred),
            "train_seconds": float(fit_seconds),
            "gpu_fallback": used_fallback,
        }

        fi = np.asarray(model_fitted.feature_importances_, dtype=np.float64)

        results.append(metrics)
        predictions[name] = (y_true, y_pred)
        importances[name] = fi

        gpu_note = " (CPU fallback)" if used_fallback else ""
        print(
            f"{name:12s} | MAE {metrics['mae']:.3f} | RMSE {metrics['rmse']:.3f} | "
            f"R2 {metrics['r2']:.4f} | sMAPE {metrics['smape']:.2f}% | {fit_seconds/60:.2f} min{gpu_note}"
        )

    results_df = pd.DataFrame(results).sort_values("mae", ascending=True).reset_index(drop=True)
    best_name = results_df.loc[0, "model"]
    y_true_best, y_pred_best = predictions[best_name]
    resid = y_pred_best - y_true_best
    abs_err = np.abs(resid)

    fi_best = importances[best_name]
    fi_series = pd.Series(fi_best, index=flat_cols).sort_values(ascending=False)
    top15 = fi_series.head(15).iloc[::-1]

    hourly_mae = []
    for h in range(24):
        mask = test_hours == h
        hourly_mae.append(abs_err[mask].mean() if mask.any() else np.nan)

    print("\nBest model:", best_name)
    print(results_df[["model", "mae", "rmse", "r2", "smape", "train_seconds", "gpu_fallback"]].to_string(index=False))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(20, 12), facecolor="#f5f4ef")
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.36, wspace=0.30)

    # 1) Forecast window (best model)
    ax_main = fig.add_subplot(gs[0, :])
    sample = min(360, len(y_true_best))
    x_axis = np.arange(sample)
    plot_true = y_true_best[-sample:]
    plot_pred = y_pred_best[-sample:]

    ax_main.set_facecolor("#ffffff")
    ax_main.plot(x_axis, plot_true, color="#0f4c81", linewidth=2.2, label="Observed")
    ax_main.plot(x_axis, plot_pred, color="#ea6a47", linewidth=2.0, linestyle="--", label=f"Predicted ({best_name})")
    ax_main.set_title("6-hour forecast window (best tree model)", fontsize=15, fontweight="bold")
    ax_main.set_ylabel("Temperature (degC)")
    ax_main.legend(loc="upper right", frameon=True)

    # 2) Top feature importances
    ax_fi = fig.add_subplot(gs[1, 0:2])
    ax_fi.set_facecolor("#ffffff")
    ax_fi.barh(top15.index, top15.values, color="#457b9d", alpha=0.92)
    ax_fi.set_title(f"Top 15 feature importances ({best_name})", fontsize=12, fontweight="bold")
    ax_fi.set_xlabel("Importance")
    ax_fi.set_ylabel("Flattened lag-feature")

    # 3) Model comparison table
    ax_table = fig.add_subplot(gs[1, 2])
    ax_table.set_facecolor("#ffffff")
    ax_table.axis("off")
    display_df = results_df[["model", "mae", "rmse", "r2", "smape"]].copy()
    for c in ["mae", "rmse", "r2", "smape"]:
        display_df[c] = display_df[c].map(lambda v: f"{v:.3f}")
    table = ax_table.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.8)
    table.scale(1.20, 1.35)
    ax_table.set_title("Model comparison", fontsize=12, fontweight="bold")

    # 4) Hourly MAE
    ax_hour = fig.add_subplot(gs[1, 3])
    ax_hour.set_facecolor("#ffffff")
    ax_hour.bar(np.arange(24), hourly_mae, color="#2a9d8f", alpha=0.9)
    ax_hour.set_title("Hourly MAE profile", fontsize=12, fontweight="bold")
    ax_hour.set_xlabel("Hour")
    ax_hour.set_ylabel("MAE (degC)")
    ax_hour.set_xticks(np.arange(0, 24, 3))

    # 5) Residual vs humidity
    ax_humid = fig.add_subplot(gs[2, 0:2])
    ax_humid.set_facecolor("#ffffff")
    ax_humid.scatter(humidity_test, resid, s=9, alpha=0.25, color="#4cc9f0")
    if len(humidity_test) > 8:
        z = np.polyfit(humidity_test, resid, deg=1)
        fit_x = np.linspace(float(humidity_test.min()), float(humidity_test.max()), 100)
        fit_y = z[0] * fit_x + z[1]
        ax_humid.plot(fit_x, fit_y, color="#d90429", linewidth=1.8, label="Trend")
        ax_humid.legend()
    ax_humid.axhline(0.0, color="#555", linestyle="--", linewidth=1.0)
    ax_humid.set_title("Residual vs humidity", fontsize=12, fontweight="bold")
    ax_humid.set_xlabel("Humidity (%)")
    ax_humid.set_ylabel("Residual (degC)")

    # 6) Residual distribution
    ax_res = fig.add_subplot(gs[2, 2:4])
    ax_res.set_facecolor("#ffffff")
    ax_res.hist(resid, bins=48, color="#f4a261", alpha=0.85, edgecolor="#1d3557")
    ax_res.axvline(0.0, color="#264653", linestyle="--", linewidth=1.3)
    ax_res.set_title("Residual distribution", fontsize=12, fontweight="bold")
    ax_res.set_xlabel("Residual (degC)")
    ax_res.set_ylabel("Count")

    best_row = results_df.iloc[0]
    fig.suptitle("Weather Tree-Ensemble Benchmark Dashboard", fontsize=18, fontweight="bold", color="#1d3557")
    fig.text(
        0.5,
        0.01,
        (
            f"Best: {best_name} | MAE {best_row['mae']:.3f} degC | RMSE {best_row['rmse']:.3f} degC | "
            f"R2 {best_row['r2']:.3f} | sMAPE {best_row['smape']:.2f}%"
        ),
        ha="center",
        fontsize=12,
        color="#1d3557",
    )
    plt.show()
