import importlib
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def ensure_dependencies():
    dependencies = {
        "lightning": "lightning>=2.1.0",
        "pytorch_forecasting": "pytorch-forecasting>=1.0.0",
    }
    for module_name, pip_name in dependencies.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            print(f"Installing missing dependency: {pip_name}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])

ensure_dependencies()

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import RMSE

CONFIG = {
    "CSV_FILE": "clean_weather_data.csv",
    "SEED": 42,
    "ENCODER_LENGTH": 48,
    "PREDICTION_LENGTH": 6,
    "TRAIN_RATIO": 0.70,
    "VAL_RATIO": 0.15,
    "BATCH_SIZE": 128,
    "MAX_EPOCHS": 40,
    "PATIENCE": 8,
    "LEARNING_RATE": 3e-4,
    "HIDDEN_SIZE": 64,
    "ATTN_HEADS": 4,
    "DROPOUT": 0.2,
    "HIDDEN_CONT_SIZE": 32,
    "NUM_WORKERS": 0,
    "USE_CUDA_IF_AVAILABLE": True,
}

TARGETS = ["temp", "pressure", "wind_speed", "visibility"]
TIME_FEATURES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos", "is_weekend"]
OBSERVED_FEATURES = ["humidity", "wind_u", "wind_v", "temp_trend", "dewpoint_approx"]

def set_seed(seed):
    pl.seed_everything(seed, workers=True)

def load_data(csv_file):
    df = pd.read_csv(csv_file)
    if "datetime" not in df.columns:
        raise ValueError("Missing required column: datetime")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    df["time_idx"] = np.arange(len(df), dtype=np.int64)
    df["group_id"] = "weather_series"
    required_base = ["temp", "pressure", "wind_speed", "visibility", "humidity", "wind_dir"]
    missing_base = [col for col in required_base if col not in df.columns]
    if missing_base:
    for col in required_base:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Build selected time features if absent.
    hour = df["datetime"].dt.hour
    dow = df["datetime"].dt.dayofweek
    doy = df["datetime"].dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.0)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.0)
    df["is_weekend"] = (dow >= 5).astype(float)

    # Build selected observed features if absent.
    wdir_rad = np.deg2rad(np.mod(df["wind_dir"], 360.0))
    df["wind_u"] = -df["wind_speed"] * np.sin(wdir_rad)
    df["wind_v"] = -df["wind_speed"] * np.cos(wdir_rad)
    df["dewpoint_approx"] = df["temp"] - ((100.0 - df["humidity"]) / 5.0)
    temp_mean_6 = df["temp"].rolling(6, min_periods=1).mean()
    temp_mean_24 = df["temp"].rolling(24, min_periods=1).mean()
    df["temp_trend"] = temp_mean_6 - temp_mean_24
    feature_cols = list(dict.fromkeys(TARGETS + TIME_FEATURES + OBSERVED_FEATURES))
    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"Missing selected feature column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[feature_cols] = df[feature_cols].interpolate(limit_direction="both").ffill().bfill()
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    df["time_idx"] = np.arange(len(df), dtype=np.int64)

    return df
def create_dataset(df, target_name, config):
    max_time_idx = int(df["time_idx"].max())
    train_cutoff = int(max_time_idx * config["TRAIN_RATIO"])
    val_cutoff = int(max_time_idx * (config["TRAIN_RATIO"] + config["VAL_RATIO"]))

    known_reals = ["time_idx"] + TIME_FEATURES
    unknown_reals = list(dict.fromkeys(TARGETS + OBSERVED_FEATURES))
    training = TimeSeriesDataSet(
        df[df["time_idx"] <= train_cutoff],
        time_idx="time_idx",
    static_categoricals=["group_id"],
    time_varying_known_reals=known_reals,
    time_varying_unknown_reals=unknown_reals,
    min_encoder_length=config["ENCODER_LENGTH"],
    max_encoder_length=config["ENCODER_LENGTH"],
    min_prediction_length=config["PREDICTION_LENGTH"],
    max_prediction_length=config["PREDICTION_LENGTH"],
    target_normalizer=None,
    allow_missing_timesteps=False,
    add_relative_time_idx=True,
    add_target_scales=False,
    add_encoder_length=True,
    )

    validation_df = df[(df["time_idx"] > train_cutoff - config["ENCODER_LENGTH"]) & (df["time_idx"] <= val_cutoff)]
    test_df = df[df["time_idx"] > val_cutoff - config["ENCODER_LENGTH"]]
    validation = TimeSeriesDataSet.from_dataset(training, validation_df, predict=True, stop_randomization=True)
    test = TimeSeriesDataSet.from_dataset(training, test_df, predict=True, stop_randomization=True)

    train_loader = training.to_dataloader(
    train=True,
    batch_size=config["BATCH_SIZE"],
    num_workers=config["NUM_WORKERS"],
    val_loader = validation.to_dataloader(
        train=False,
        batch_size=config["BATCH_SIZE"],
        num_workers=config["NUM_WORKERS"],
    )

    return {
    "training": training,
    "validation": validation,
    "test": test,
    "train_loader": train_loader,
    "val_loader": val_loader,
    "test_loader": test_loader,
    }

def train_model(training_dataset, train_loader, val_loader, config, target_name):
    tft = TemporalFusionTransformer.from_dataset(
    learning_rate=config["LEARNING_RATE"],
    hidden_size=config["HIDDEN_SIZE"],
    attention_head_size=config["ATTN_HEADS"],
    dropout=config["DROPOUT"],
    hidden_continuous_size=config["HIDDEN_CONT_SIZE"],
    loss=RMSE(),
        log_interval=10,
        reduce_on_plateau_patience=3,
    )
    early_stop = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=config["PATIENCE"], mode="min")
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    checkpoint = ModelCheckpoint(
    monitor="val_loss",
    mode="min",
    save_top_k=1,
        filename=f"tft_{target_name}" + "-{epoch:02d}-{val_loss:.4f}",
    )

    use_cuda = config["USE_CUDA_IF_AVAILABLE"] and torch.cuda.is_available()
    trainer = pl.Trainer(
        max_epochs=config["MAX_EPOCHS"],
        accelerator="gpu" if use_cuda else "cpu",
    devices=1,
    gradient_clip_val=0.1,
    callbacks=[early_stop, lr_monitor, checkpoint],
        logger=False,
        enable_model_summary=True,
    )
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_path = checkpoint.best_model_path
    if best_path:
        best_model = TemporalFusionTransformer.load_from_checkpoint(best_path)
    else:
        best_model = tft

    return best_model

def evaluate_model(model, test_loader):
    predictions, x = model.predict(test_loader, mode="prediction", return_x=True)

    y_pred = predictions.detach().cpu().numpy().reshape(-1)
    y_true = x["decoder_target"].detach().cpu().numpy().reshape(-1)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan")
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }

def main():
    warnings.filterwarnings("ignore")
    set_seed(CONFIG["SEED"])

    df = load_data(CONFIG["CSV_FILE"])
    print(
        "Data loaded | rows={} | range={} -> {}".format(
            len(df),
            df["datetime"].min(),
            df["datetime"].max(),
        )
    )
    print(
        "Setup | encoder_len={} prediction_len={} targets={}".format(
            CONFIG["ENCODER_LENGTH"],
            CONFIG["PREDICTION_LENGTH"],
            TARGETS,
        )
    )

    all_metrics = {}
    for target_name in TARGETS:
        print("\n" + "=" * 72)
        print(f"Training TFT for target: {target_name}")
        dataset_bundle = create_dataset(df, target_name, CONFIG)
        model = train_model(
            training_dataset=dataset_bundle["training"],
            train_loader=dataset_bundle["train_loader"],
            val_loader=dataset_bundle["val_loader"],
            config=CONFIG,
            target_name=target_name,
        )
        metrics = evaluate_model(model, dataset_bundle["test_loader"])
        all_metrics[target_name] = metrics
        print(
            f"TEST {target_name:10s} | MAE={metrics['MAE']:.4f} "
            f"RMSE={metrics['RMSE']:.4f} R2={metrics['R2']:.4f}"
        )

    print("\n" + "=" * 72)
    print("Per-target TFT test metrics")
    for target_name in TARGETS:
        m = all_metrics[target_name]
        print(f"{target_name:10s} | MAE={m['MAE']:.4f} RMSE={m['RMSE']:.4f} R2={m['R2']:.4f}")

    avg_mae = float(np.mean([m["MAE"] for m in all_metrics.values()]))
    avg_rmse = float(np.mean([m["RMSE"] for m in all_metrics.values()]))
    avg_r2 = float(np.mean([m["R2"] for m in all_metrics.values()]))
    print("\nAverage metrics across targets")
    print(f"MAE={avg_mae:.4f} RMSE={avg_rmse:.4f} R2={avg_r2:.4f}")

if __name__ == "__main__":
    main()
import random
import time

import numpy as np
import pandas as pd
import xgboost as xgb
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor
from xgboost.callback import EarlyStopping

try:
    import optuna
except ImportError:
    optuna = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


CONFIG = {
    "CSV_FILE": "clean_weather_data.csv",
    "SEED": 42,
    "SEQ_LENGTH": 48,
    "PRED_STEP": 6,
    "TRAIN_RATIO": 0.70,
    "VAL_RATIO": 0.15,
    "BATCH_SIZE": 64,
    "EPOCHS": 150,
    "PATIENCE": 20,
    "LR": 3e-4,
    "WEIGHT_DECAY": 3e-3,
    "HIDDEN": 192,
    "LAYERS": 2,
    "DROPOUT": 0.35,
    "USE_CUDA_IF_AVAILABLE": True,
    "XGB_TRIALS": 30,
    "USE_ATTENTION": True,
    "SHOW_PLOTS": True,
}

TARGET_VARS = ["temp", "pressure", "wind_speed", "visibility"]
TARGET_INDICES = [0, 1, 2, 3]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def progress(iterable, desc):
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, leave=False)


def clean_numeric(df, cols):
    df = df.copy()
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].interpolate(limit_direction="both").ffill().bfill()
    return df


def load_and_engineer(file_path):
    df = pd.read_csv(file_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    required = ["temp", "pressure", "wind_speed", "visibility", "humidity", "wind_dir"]
    for c in required:
        if c not in df.columns:
            if c == "humidity":
                df[c] = 60.0
            elif c == "wind_dir":
                df[c] = 0.0
            else:
                raise ValueError(f"Missing required column: {c}")

    excluded_cols = {"datetime"}
    extra_numeric = []
    for col in df.columns:
        if col in excluded_cols or col in required:
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() > 0:
            df[col] = converted
            extra_numeric.append(col)

    df = clean_numeric(df, required + extra_numeric)
    df["wind_dir"] = np.mod(df["wind_dir"], 360.0)

    # Time features.
    hr = df["datetime"].dt.hour
    dow = df["datetime"].dt.dayofweek
    doy = df["datetime"].dt.dayofyear

    new_cols = {
        "hour_sin": np.sin(2 * np.pi * hr / 24.0),
        "hour_cos": np.cos(2 * np.pi * hr / 24.0),
        "dow_sin": np.sin(2 * np.pi * dow / 7.0),
        "dow_cos": np.cos(2 * np.pi * dow / 7.0),
        "doy_sin": np.sin(2 * np.pi * doy / 365.0),
        "doy_cos": np.cos(2 * np.pi * doy / 365.0),
        "is_weekend": (dow >= 5).astype(float),
    }

    # Wind direction circular/component features.
    wdir_rad = np.deg2rad(df["wind_dir"])
    new_cols["wind_sin"] = np.sin(np.deg2rad(df["wind_dir"].fillna(0.0)))
    new_cols["wind_cos"] = np.cos(np.deg2rad(df["wind_dir"].fillna(0.0)))
    new_cols["wind_u"] = -df["wind_speed"] * np.sin(wdir_rad)
    new_cols["wind_v"] = -df["wind_speed"] * np.cos(wdir_rad)

    # Physical interaction features.
    new_cols["dewpoint_approx"] = df["temp"] - ((100 - df["humidity"]) / 5.0)
    new_cols["apparent_temp"] = df["temp"] - 0.4 * (df["temp"] - 10) * (1 - df["humidity"] / 100)
    new_cols["temp_humidity_interact"] = df["temp"] * (df["humidity"] / 100.0)
    new_cols["pressure_trend_3h"] = df["pressure"].diff(3).fillna(0.0)
    new_cols["pressure_trend_12h"] = df["pressure"].diff(12).fillna(0.0)
    new_cols["wind_speed_sq"] = df["wind_speed"] ** 2

    # Per-variable feature engineering: rolling/lag/diff for ALL targets.
    for col in TARGET_VARS:
        new_cols[f"{col}_mean_6"] = df[col].rolling(6, min_periods=1).mean()
        new_cols[f"{col}_mean_24"] = df[col].rolling(24, min_periods=1).mean()
        new_cols[f"{col}_mean_48"] = df[col].rolling(48, min_periods=1).mean()
        new_cols[f"{col}_q10_24"] = df[col].rolling(24, min_periods=1).quantile(0.10)
        new_cols[f"{col}_q90_24"] = df[col].rolling(24, min_periods=1).quantile(0.90)
        new_cols[f"{col}_std_6"] = df[col].rolling(6, min_periods=1).std().fillna(0.0)
        new_cols[f"{col}_diff_1"] = df[col].diff(1).fillna(0.0)
        new_cols[f"{col}_diff_6"] = df[col].diff(6).fillna(0.0)
        if col in ["temp", "pressure"]:
            lag_list = [1, 2, 3, 6, 12, 24, 48, 72, 168]
        else:
            lag_list = [1, 2, 3, 6, 12, 24, 48]
        for lag in lag_list:
            new_cols[f"{col}_lag_{lag}"] = df[col].shift(lag)

    # Extra trend signal for temperature.
    new_cols["temp_trend"] = new_cols["temp_mean_6"] - new_cols["temp_mean_24"]

    # Use additional observed weather fields from CSV (if available) as predictors.
    aux_feature_vars = [c for c in extra_numeric if c not in TARGET_VARS and c not in ["humidity", "wind_dir"]]
    for col in aux_feature_vars:
        new_cols[f"{col}_mean_6"] = df[col].rolling(6, min_periods=1).mean()
        new_cols[f"{col}_mean_24"] = df[col].rolling(24, min_periods=1).mean()
        new_cols[f"{col}_diff_1"] = df[col].diff(1).fillna(0.0)
        new_cols[f"{col}_lag_1"] = df[col].shift(1)
        new_cols[f"{col}_lag_6"] = df[col].shift(6)

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    df = df.copy()
    df = df.ffill().bfill()

    feature_cols = [
        "temp",
        "pressure",
        "wind_speed",
        "visibility",
        "humidity",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "doy_sin",
        "doy_cos",
        "is_weekend",
        "wind_sin",
        "wind_cos",
        "wind_u",
        "wind_v",
        "temp_trend",
        "dewpoint_approx",
        "apparent_temp",
        "temp_humidity_interact",
        "pressure_trend_3h",
        "pressure_trend_12h",
        "wind_speed_sq",
    ]

    for col in TARGET_VARS:
        col_features = [
            f"{col}_mean_6",
            f"{col}_mean_24",
            f"{col}_mean_48",
            f"{col}_q10_24",
            f"{col}_q90_24",
            f"{col}_std_6",
            f"{col}_diff_1",
            f"{col}_diff_6",
            f"{col}_lag_1",
            f"{col}_lag_2",
            f"{col}_lag_3",
            f"{col}_lag_6",
            f"{col}_lag_12",
            f"{col}_lag_24",
            f"{col}_lag_48",
        ]
        if col in ["temp", "pressure"]:
            col_features += [f"{col}_lag_72", f"{col}_lag_168"]
        feature_cols += col_features

    for col in aux_feature_vars:
        feature_cols += [
            col,
            f"{col}_mean_6",
            f"{col}_mean_24",
            f"{col}_diff_1",
            f"{col}_lag_1",
            f"{col}_lag_6",
        ]

    feature_cols = list(dict.fromkeys(feature_cols))
    if aux_feature_vars:
        print(f"Using additional CSV variables: {aux_feature_vars}")

    return df, feature_cols


def create_sequences(x_scaled, y_scaled, seq_len, pred_step):
    x_seq, y_seq, t_idx = [], [], []
    max_start = len(x_scaled) - seq_len - pred_step + 1
    for i in range(max_start):
        idx = i + seq_len + pred_step - 1
        x_seq.append(x_scaled[i : i + seq_len])
        y_seq.append(y_scaled[idx])
        t_idx.append(idx)
    return (
        np.asarray(x_seq, dtype=np.float32),
        np.asarray(y_seq, dtype=np.float32),
        np.asarray(t_idx, dtype=np.int64),
    )


def inverse_all(scaler, arr):
    return scaler.inverse_transform(arr)


def predict_sequences(model, seq_array, device):
    loader = DataLoader(
        TensorDataset(torch.from_numpy(seq_array)),
        batch_size=CONFIG["BATCH_SIZE"],
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )
    preds = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                out = model(xb)
            preds.append(out.float().cpu().numpy())
    return np.concatenate(preds, axis=0)


def build_target_feature_map(feature_cols):
    return {var: feature_cols.index(var) for var in TARGET_VARS}


def make_xgb_features(x_seq):
    last = x_seq[:, -1, :]
    first = x_seq[:, 0, :]
    mean_all = x_seq.mean(axis=1)
    std_all = x_seq.std(axis=1)
    min_all = x_seq.min(axis=1)
    max_all = x_seq.max(axis=1)

    # Temporal contrast features: recent segment vs older context (non-overlapping).
    seq_len = x_seq.shape[1]
    recent_k = max(1, seq_len // 4)
    recent_start = max(0, seq_len - recent_k)
    older_end = max(1, recent_start)
    recent_mean = x_seq[:, recent_start:, :].mean(axis=1)
    older_mean = x_seq[:, :older_end, :].mean(axis=1)
    recent_trend = recent_mean - older_mean
    slope = (last - first) / float(max(1, seq_len))

    return np.concatenate(
        [last, mean_all, std_all, min_all, max_all, recent_mean, older_mean, recent_trend, slope],
        axis=1,
    )


class HybridMSEMAELoss(nn.Module):
    def __init__(self, mse_weight=0.7, mae_weight=0.3):
        super().__init__()
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, pred, target):
        return self.mse_weight * self.mse(pred, target) + self.mae_weight * self.mae(pred, target)


class AttentionWeatherLSTM(nn.Module):
    def __init__(self, in_dim, hidden, layers, dropout, out_dim, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        self.input_proj = nn.Linear(in_dim, hidden)
        self.lstm = nn.LSTM(
            hidden,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=True,
        )
        self.attn = nn.Linear(hidden * 2, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        x = self.input_proj(x)
        seq_out, _ = self.lstm(x)
        if self.use_attention:
            attn_w = torch.softmax(self.attn(seq_out), dim=1)
            context = (seq_out * attn_w).sum(dim=1)
        else:
            context = seq_out[:, -1, :]
        return self.head(context)


def detect_xgb_runtime(seed, use_cuda=True):
    if not use_cuda:
        return {"tree_method": "hist", "n_jobs": -1}, "cpu"

    x_probe = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    y_probe = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    candidates = [
        {"tree_method": "hist", "device": "cuda"},
        {"tree_method": "gpu_hist", "predictor": "gpu_predictor"},
    ]
    for cfg in candidates:
        try:
            m = XGBRegressor(
                n_estimators=2,
                max_depth=2,
                learning_rate=0.1,
                objective="reg:squarederror",
                eval_metric="rmse",
                random_state=seed,
                **cfg,
            )
            m.fit(x_probe, y_probe, verbose=False)
            return cfg, "cuda"
        except Exception:
            continue
    return {"tree_method": "hist", "n_jobs": -1}, "cpu"


def sample_xgb_params(n_trials, seed, runtime_params):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_trials):
        p = {
            "n_estimators": int(rng.integers(200, 500)),
            "max_depth": int(rng.integers(3, 11)),
            "learning_rate": float(rng.uniform(0.005, 0.1)),
            "subsample": float(rng.uniform(0.6, 0.95)),
            "colsample_bytree": float(rng.uniform(0.6, 0.95)),
            "min_child_weight": float(rng.uniform(1.0, 10.0)),
            "reg_alpha": float(rng.uniform(0.0, 0.5)),
            "reg_lambda": float(rng.uniform(0.5, 3.0)),
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "random_state": seed,
        }
        p.update(runtime_params)
        out.append(p)
    return out


def train_xgb_booster(params, x_train_flat, y_train, x_val_flat=None, y_val=None, early_stop_rounds=None):
    train_matrix = xgb.DMatrix(x_train_flat, label=y_train)
    evals = None
    if x_val_flat is not None and y_val is not None:
        evals = [(train_matrix, "train"), (xgb.DMatrix(x_val_flat, label=y_val), "val")]

    train_params = dict(params)
    num_boost_round = int(train_params.pop("n_estimators", 300))
    callbacks = None
    if early_stop_rounds and evals is not None:
        callbacks = [xgb.callback.EarlyStopping(rounds=early_stop_rounds, save_best=True)]

    booster = xgb.train(
        train_params,
        train_matrix,
        num_boost_round=num_boost_round,
        evals=evals,
        verbose_eval=False,
        callbacks=callbacks,
    )
    return booster


def predict_xgb_booster(booster, x_flat):
    dmatrix = xgb.DMatrix(x_flat)
    best_iter = getattr(booster, "best_iteration", None)
    if best_iter is not None and best_iter >= 0:
        return booster.predict(dmatrix, iteration_range=(0, best_iter + 1))
    return booster.predict(dmatrix)


def make_lstm_loaders(x_train_seq, y_train, x_val_seq, y_val):
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train_seq), torch.from_numpy(y_train)),
        batch_size=CONFIG["BATCH_SIZE"],
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val_seq), torch.from_numpy(y_val)),
        batch_size=CONFIG["BATCH_SIZE"],
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def train_attention_lstm(x_train_seq, y_train, x_val_seq, y_val, in_dim, out_dim, device):
    model = AttentionWeatherLSTM(
        in_dim=in_dim,
        hidden=CONFIG["HIDDEN"],
        layers=CONFIG["LAYERS"],
        dropout=CONFIG["DROPOUT"],
        out_dim=out_dim,
        use_attention=CONFIG.get("USE_ATTENTION", True),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["LR"],
        weight_decay=CONFIG["WEIGHT_DECAY"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,
        T_mult=2,
        eta_min=1e-6,
    )
    criterion = HybridMSEMAELoss(mse_weight=0.7, mae_weight=0.3)
    scaler_amp = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    train_loader, val_loader = make_lstm_loaders(x_train_seq, y_train, x_val_seq, y_val)

    best_state = None
    best_val = float("inf")
    best_epoch = 1
    stale = 0
    hist_train, hist_val = [], []

    for epoch in range(CONFIG["EPOCHS"]):
        model.train()
        tr_sum = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(xb)
                loss = criterion(pred, yb)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            tr_sum += loss.item()

        model.eval()
        va_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    pred = model(xb)
                    loss = criterion(pred, yb)
                va_sum += loss.item()

        tr_loss = tr_sum / max(1, len(train_loader))
        va_loss = va_sum / max(1, len(val_loader))
        hist_train.append(tr_loss)
        hist_val.append(va_loss)
        scheduler.step(epoch + 1)

        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  LSTM epoch {epoch+1:3d}/{CONFIG['EPOCHS']} | train={tr_loss:.5f} val={va_loss:.5f}")

        if stale >= CONFIG["PATIENCE"]:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, hist_train, hist_val


def train_lstm_final(x_train_full_seq, y_train_full, in_dim, out_dim, device, epochs):
    model = AttentionWeatherLSTM(
        in_dim=in_dim,
        hidden=CONFIG["HIDDEN"],
        layers=CONFIG["LAYERS"],
        dropout=CONFIG["DROPOUT"],
        out_dim=out_dim,
        use_attention=CONFIG.get("USE_ATTENTION", True),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["LR"],
        weight_decay=CONFIG["WEIGHT_DECAY"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=1e-5,
    )
    criterion = HybridMSEMAELoss(mse_weight=0.7, mae_weight=0.3)
    scaler_amp = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train_full_seq), torch.from_numpy(y_train_full)),
        batch_size=CONFIG["BATCH_SIZE"],
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )

    for epoch in range(max(1, epochs)):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(xb)
                loss = criterion(pred, yb)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
        scheduler.step()
    return model


def tune_xgb_target(x_train_flat, y_train, x_val_flat, y_val, runtime_params, target_name):
    def run_random_search():
        param_sets = sample_xgb_params(CONFIG["XGB_TRIALS"], CONFIG["SEED"], runtime_params)
        best_model = None
        best_params = None
        best_rmse = float("inf")
        best_pred_val = None

        print(f"  Tuning XGB for {target_name} ({len(param_sets)} random trials)...")
        for i, params in enumerate(progress(param_sets, f"XGB {target_name}"), 1):
            model = train_xgb_booster(
                params,
                x_train_flat=x_train_flat,
                y_train=y_train,
                x_val_flat=x_val_flat,
                y_val=y_val,
                early_stop_rounds=30,
            )
            pred_val = predict_xgb_booster(model, x_val_flat)
            rmse = float(np.sqrt(mean_squared_error(y_val, pred_val)))
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                best_params = params
                best_pred_val = pred_val

            if i % 5 == 0 or i == len(param_sets):
                print(f"    trial {i:2d}/{len(param_sets)} | best val RMSE={best_rmse:.5f}")

        best_iter = getattr(best_model, "best_iteration", None)
        if best_iter is None or best_iter <= 0:
            best_iter = best_params.get("n_estimators", 300)

        return {
            "model": best_model,
            "params": best_params,
            "pred_val": best_pred_val,
            "best_iter": int(best_iter),
        }

    if optuna is None:
        print("  Warning: Optuna not installed. Falling back to random XGB search.")
        return run_random_search()

    try:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=CONFIG["SEED"])
        study = optuna.create_study(direction="minimize", sampler=sampler)

        print(f"  Tuning XGB for {target_name} ({CONFIG['XGB_TRIALS']} Optuna TPE trials)...")

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 600),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.12, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
                "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
                "gamma": trial.suggest_float("gamma", 0.0, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 4.0),
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "random_state": CONFIG["SEED"],
            }
            params.update(runtime_params)
            model = train_xgb_booster(
                params,
                x_train_flat=x_train_flat,
                y_train=y_train,
                x_val_flat=x_val_flat,
                y_val=y_val,
                early_stop_rounds=30,
            )
            pred_val = predict_xgb_booster(model, x_val_flat)
            rmse = float(np.sqrt(mean_squared_error(y_val, pred_val)))
            trial.set_user_attr("best_iteration", int(getattr(model, "best_iteration", -1)))
            return rmse

        study.optimize(objective, n_trials=CONFIG["XGB_TRIALS"], show_progress_bar=False)

        best_params = {
            "n_estimators": int(study.best_params["n_estimators"]),
            "max_depth": int(study.best_params["max_depth"]),
            "learning_rate": float(study.best_params["learning_rate"]),
            "subsample": float(study.best_params["subsample"]),
            "colsample_bytree": float(study.best_params["colsample_bytree"]),
            "colsample_bylevel": float(study.best_params["colsample_bylevel"]),
            "min_child_weight": float(study.best_params["min_child_weight"]),
            "gamma": float(study.best_params["gamma"]),
            "reg_alpha": float(study.best_params["reg_alpha"]),
            "reg_lambda": float(study.best_params["reg_lambda"]),
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "random_state": CONFIG["SEED"],
        }
        best_params.update(runtime_params)

        best_model = train_xgb_booster(
            best_params,
            x_train_flat=x_train_flat,
            y_train=y_train,
            x_val_flat=x_val_flat,
            y_val=y_val,
            early_stop_rounds=30,
        )
        best_pred_val = predict_xgb_booster(best_model, x_val_flat)
        best_iter = getattr(best_model, "best_iteration", None)
        if best_iter is None or best_iter <= 0:
            best_iter = best_params.get("n_estimators", 300)
        print(f"    best Optuna val RMSE={study.best_value:.5f}")

        return {
            "model": best_model,
            "params": best_params,
            "pred_val": best_pred_val,
            "best_iter": int(best_iter),
        }
    except Exception as exc:
        print(f"  Warning: Optuna tuning failed ({exc}). Falling back to random XGB search.")
        return run_random_search()


def refit_xgb(best_params, x_train_full_flat, y_train_full):
    params = dict(best_params)
    params.pop("early_stopping_rounds", None)
    return train_xgb_booster(params, x_train_full_flat, y_train_full)


def train_meta_model(meta_x, meta_y, seed, target_name):
    if lgb is not None:
        try:
            meta = lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.02,
                num_leaves=15,
                min_child_samples=30,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                verbose=-1,
            )
            meta.fit(meta_x, meta_y)
            return meta, "lightgbm"
        except Exception:
            pass
    meta = Ridge(alpha=0.5)
    meta.fit(meta_x, meta_y)
    return meta, "ridge"


def print_metrics_row(prefix, m):
    print(
        f"{prefix:16s} | "
        f"MAE {m['MAE']:.3f} | RMSE {m['RMSE']:.3f} | R2 {m['R2']:.4f}"
    )


def metric_row(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


if __name__ == "__main__":
    set_seed(CONFIG["SEED"])
    t0 = time.time()

    df, feature_cols = load_and_engineer(CONFIG["CSV_FILE"])
    x_raw = df[feature_cols].values
    y_raw = df[TARGET_VARS].values

    n = len(x_raw)
    train_cut = int(n * CONFIG["TRAIN_RATIO"])
    val_cut = int(n * (CONFIG["TRAIN_RATIO"] + CONFIG["VAL_RATIO"]))

    x_scaler = RobustScaler()
    y_scaler = RobustScaler()
    x_scaler.fit(x_raw[:train_cut])
    y_scaler.fit(y_raw[:train_cut])

    x_scaled = x_scaler.transform(x_raw)
    y_scaled = y_scaler.transform(y_raw)

    x_seq, y_seq, t_idx = create_sequences(
        x_scaled,
        y_scaled,
        CONFIG["SEQ_LENGTH"],
        CONFIG["PRED_STEP"],
    )

    train_mask = t_idx < train_cut
    val_mask = (t_idx >= train_cut) & (t_idx < val_cut)
    test_mask = t_idx >= val_cut

    x_train_seq, y_train = x_seq[train_mask], y_seq[train_mask]
    x_val_seq, y_val = x_seq[val_mask], y_seq[val_mask]
    x_test_seq, y_test = x_seq[test_mask], y_seq[test_mask]

    x_train_flat = make_xgb_features(x_train_seq)
    x_val_flat = make_xgb_features(x_val_seq)
    x_test_flat = make_xgb_features(x_test_seq)
    target_feat_idx = build_target_feature_map(feature_cols)

    x_train_full_seq = np.concatenate([x_train_seq, x_val_seq], axis=0)
    y_train_full = np.concatenate([y_train, y_val], axis=0)
    x_train_full_flat = np.concatenate([x_train_flat, x_val_flat], axis=0)

    device = torch.device("cuda" if (CONFIG["USE_CUDA_IF_AVAILABLE"] and torch.cuda.is_available()) else "cpu")
    runtime_params, runtime_mode = detect_xgb_runtime(CONFIG["SEED"], CONFIG["USE_CUDA_IF_AVAILABLE"])

    print(f"LSTM device: {device}")
    print(f"XGB backend: {runtime_mode} | runtime params: {runtime_params}")
    print(f"Attention enabled: {CONFIG.get('USE_ATTENTION', True)}")
    print(
        f"Samples | train={len(x_train_seq)} val={len(x_val_seq)} test={len(x_test_seq)} | "
        f"seq_len={CONFIG['SEQ_LENGTH']} pred_step={CONFIG['PRED_STEP']}"
    )

    # Train attention-LSTM base model.
    lstm_base, best_epoch, lstm_train_hist, lstm_val_hist = train_attention_lstm(
        x_train_seq=x_train_seq,
        y_train=y_train,
        x_val_seq=x_val_seq,
        y_val=y_val,
        in_dim=x_train_seq.shape[-1],
        out_dim=len(TARGET_VARS),
        device=device,
    )

    # Final LSTM refit on train+val for test predictions.
    lstm_final = train_lstm_final(
        x_train_full_seq=x_train_full_seq,
        y_train_full=y_train_full,
        in_dim=x_train_seq.shape[-1],
        out_dim=len(TARGET_VARS),
        device=device,
        epochs=best_epoch,
    )

    # XGB per target with tuning.
    xgb_val_scaled = np.zeros_like(y_val)
    xgb_test_scaled = np.zeros_like(y_test)
    xgb_best = {}
    meta_split_idx = len(y_val) // 2
    if meta_split_idx <= 0 or meta_split_idx >= len(y_val):
        meta_split_idx = max(1, len(y_val) - 1)

    meta_train_slice = slice(0, meta_split_idx)
    meta_val_slice = slice(meta_split_idx, len(y_val))
    if meta_split_idx >= len(y_val):
        meta_val_slice = slice(0, len(y_val))

    for j, target in enumerate(TARGET_VARS):
        tuned = tune_xgb_target(
            x_train_flat=x_train_flat,
            y_train=y_train[:, j],
            x_val_flat=x_val_flat,
            y_val=y_val[:, j],
            runtime_params=runtime_params,
            target_name=target,
        )
        xgb_best[target] = tuned
        xgb_val_scaled[:, j] = tuned["pred_val"]

        final_params = dict(tuned["params"])
        final_params.pop("early_stopping_rounds", None)
        best_iter = tuned["best_iter"]
        if best_iter is not None and best_iter > 0:
            final_params["n_estimators"] = int(max(50, best_iter * 1.1))

        final_model = train_xgb_booster(final_params, x_train_full_flat, y_train_full[:, j])
        xgb_test_scaled[:, j] = predict_xgb_booster(final_model, x_test_flat)
        xgb_best[target]["final_model"] = final_model

    # LSTM predictions.
    lstm_val_scaled = predict_sequences(lstm_base, x_val_seq, device)
    lstm_test_scaled = predict_sequences(lstm_final, x_test_seq, device)

    # Meta stacking per variable.
    stack_val_candidate_scaled = np.zeros_like(y_val)
    stack_test_candidate_scaled = np.zeros_like(y_test)
    stack_val_scaled = np.zeros_like(y_val)
    stack_test_scaled = np.zeros_like(y_test)
    stack_meta_val_candidate_scaled = np.zeros((len(y_val[meta_val_slice]), y_val.shape[1]), dtype=np.float32)
    xgb_meta_val_scaled = np.zeros((len(y_val[meta_val_slice]), y_val.shape[1]), dtype=np.float32)
    y_meta_val_scaled = y_val[meta_val_slice]
    meta_models = {}
    passthrough_vars = []
    full_stacking_vars = []

    for j, target in enumerate(TARGET_VARS):
        feat_idx = target_feat_idx[target]
        last_obs_val = x_val_seq[:, -1, feat_idx]
        last_obs_test = x_test_seq[:, -1, feat_idx]
        lag6_idx = -7 if x_val_seq.shape[1] >= 7 else 0
        last_obs_lag6_val = x_val_seq[:, lag6_idx, feat_idx]
        last_obs_lag6_test = x_test_seq[:, lag6_idx, feat_idx]
        obs_trend_val = last_obs_val - last_obs_lag6_val
        obs_trend_test = last_obs_test - last_obs_lag6_test
        lstm_error_val = last_obs_val - lstm_val_scaled[:, j]
        xgb_error_val = last_obs_val - xgb_val_scaled[:, j]
        lstm_error_test = last_obs_test - lstm_test_scaled[:, j]
        xgb_error_test = last_obs_test - xgb_test_scaled[:, j]
        pred_spread_val = np.abs(lstm_val_scaled[:, j] - xgb_val_scaled[:, j])
        pred_spread_test = np.abs(lstm_test_scaled[:, j] - xgb_test_scaled[:, j])
        error_agree_val = lstm_error_val * xgb_error_val
        error_agree_test = lstm_error_test * xgb_error_test
        meta_cols = [
            "lstm_pred",
            "xgb_pred",
            "last_obs",
            "lstm_error",
            "xgb_error",
            "pred_spread",
            "error_agree",
            "last_obs_lag6",
            "obs_trend",
        ]
        meta_x_val_full = pd.DataFrame(np.column_stack([
            lstm_val_scaled[:, j],
            xgb_val_scaled[:, j],
            last_obs_val,
            lstm_error_val,
            xgb_error_val,
            pred_spread_val,
            error_agree_val,
            last_obs_lag6_val,
            obs_trend_val,
        ]), columns=meta_cols)
        meta_x_test = pd.DataFrame(np.column_stack([
            lstm_test_scaled[:, j],
            xgb_test_scaled[:, j],
            last_obs_test,
            lstm_error_test,
            xgb_error_test,
            pred_spread_test,
            error_agree_test,
            last_obs_lag6_test,
            obs_trend_test,
        ]), columns=meta_cols)
        meta_x_train = meta_x_val_full.iloc[meta_train_slice]
        y_meta_train = y_val[:, j][meta_train_slice]
        meta_x_guard = meta_x_val_full.iloc[meta_val_slice]
        y_meta_guard = y_val[:, j][meta_val_slice]
        if len(meta_x_guard) == 0:
            meta_x_guard = meta_x_val_full
            y_meta_guard = y_val[:, j]

        meta, meta_kind = train_meta_model(meta_x_train, y_meta_train, CONFIG["SEED"], target)
        stack_val_pred = meta.predict(meta_x_val_full)
        stack_meta_guard_pred = meta.predict(meta_x_guard)
        stack_test_pred = meta.predict(meta_x_test)
        stack_val_candidate_scaled[:, j] = stack_val_pred
        stack_test_candidate_scaled[:, j] = stack_test_pred
        stack_meta_val_candidate_scaled[:, j] = stack_meta_guard_pred
        xgb_meta_val_scaled[:, j] = xgb_val_scaled[:, j][meta_val_slice] if len(y_meta_guard) == len(y_val[:, j][meta_val_slice]) else xgb_val_scaled[:, j]
        meta_models[target] = {"model": meta, "kind": meta_kind}

    # Inverse transform for reporting.
    y_val_inv = inverse_all(y_scaler, y_val)
    y_test_inv = inverse_all(y_scaler, y_test)
    lstm_val_inv = inverse_all(y_scaler, lstm_val_scaled)
    lstm_test_inv = inverse_all(y_scaler, lstm_test_scaled)
    xgb_val_inv = inverse_all(y_scaler, xgb_val_scaled)
    xgb_test_inv = inverse_all(y_scaler, xgb_test_scaled)
    stack_val_candidate_inv = inverse_all(y_scaler, stack_val_candidate_scaled)
    stack_test_candidate_inv = inverse_all(y_scaler, stack_test_candidate_scaled)

    # Passthrough guard must compare validation RMSE on original (inverse-transformed) scale.
    y_meta_val_inv = inverse_all(y_scaler, y_meta_val_scaled)
    stack_meta_val_candidate_inv = inverse_all(y_scaler, stack_meta_val_candidate_scaled)
    xgb_meta_val_inv = inverse_all(y_scaler, xgb_meta_val_scaled)

    xgb_val_rmse = {
        var: float(np.sqrt(mean_squared_error(y_meta_val_inv[:, j], xgb_meta_val_inv[:, j])))
        for j, var in enumerate(TARGET_VARS)
    }
    stack_val_rmse = {
        var: float(np.sqrt(mean_squared_error(y_meta_val_inv[:, j], stack_meta_val_candidate_inv[:, j])))
        for j, var in enumerate(TARGET_VARS)
    }
    print(f"Validation RMSE (original scale) | XGB: {xgb_val_rmse}")
    print(f"Validation RMSE (original scale) | STACK: {stack_val_rmse}")

    for j, target in enumerate(TARGET_VARS):
        if stack_val_rmse[target] > xgb_val_rmse[target]:
            stack_val_scaled[:, j] = xgb_val_scaled[:, j]
            stack_test_scaled[:, j] = xgb_test_scaled[:, j]
            passthrough_vars.append(target)
            meta_models[target] = {"model": None, "kind": "xgb_passthrough"}
            print(
                f"  Stack passthrough {target}: stack val RMSE {stack_val_rmse[target]:.5f} > "
                f"xgb val RMSE {xgb_val_rmse[target]:.5f}"
            )
        else:
            stack_val_scaled[:, j] = stack_val_candidate_scaled[:, j]
            stack_test_scaled[:, j] = stack_test_candidate_scaled[:, j]
            full_stacking_vars.append(target)

    stack_val_inv = inverse_all(y_scaler, stack_val_scaled)
    stack_test_inv = inverse_all(y_scaler, stack_test_scaled)

    print("\nPer-variable TEST metrics")
    print("variable    | model   | MAE     RMSE    R2")
    per_model_metrics = {"LSTM": [], "XGB": [], "STACK": []}
    for j, var in enumerate(TARGET_VARS):
        m_l = metric_row(y_test_inv[:, j], lstm_test_inv[:, j])
        m_x = metric_row(y_test_inv[:, j], xgb_test_inv[:, j])
        m_s = metric_row(y_test_inv[:, j], stack_test_inv[:, j])
        per_model_metrics["LSTM"].append(m_l)
        per_model_metrics["XGB"].append(m_x)
        per_model_metrics["STACK"].append(m_s)
        print(f"{var:11s}| LSTM   | {m_l['MAE']:.3f}  {m_l['RMSE']:.3f}  {m_l['R2']:.4f}")
        print(f"{var:11s}| XGB    | {m_x['MAE']:.3f}  {m_x['RMSE']:.3f}  {m_x['R2']:.4f}")
        print(f"{var:11s}| STACK  | {m_s['MAE']:.3f}  {m_s['RMSE']:.3f}  {m_s['R2']:.4f}")

    avg_r2_l = float(np.mean([m["R2"] for m in per_model_metrics["LSTM"]]))
    avg_r2_x = float(np.mean([m["R2"] for m in per_model_metrics["XGB"]]))
    avg_r2_s = float(np.mean([m["R2"] for m in per_model_metrics["STACK"]]))

    print("\nAverage TEST R2")
    print(f"LSTM : {avg_r2_l:.4f}")
    print(f"XGB  : {avg_r2_x:.4f}")
    print(f"STACK: {avg_r2_s:.4f}")
    print(
        "Stacking summary | "
        f"XGB passthrough: {passthrough_vars if passthrough_vars else ['none']} | "
        f"Full stacking: {full_stacking_vars if full_stacking_vars else ['none']}"
    )
    print(f"Total runtime: {(time.time() - t0) / 60:.2f} min")

    if CONFIG["SHOW_PLOTS"]:
        # Optional diagnostic plot: stacked predictions vs truth.
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
        axes = axes.flatten()
        sample = min(240, len(y_test_inv))
        for j, var in enumerate(TARGET_VARS):
            axes[j].plot(y_test_inv[-sample:, j], label="Actual", lw=2)
            axes[j].plot(stack_test_inv[-sample:, j], label="Stacked", lw=1.7, ls="--")
            axes[j].set_title(var)
            axes[j].legend()
        fig.suptitle("Stacked Forecasts by Variable")
        plt.show()
