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
        raise ValueError(f"Missing required weather columns: {missing_base}")

    for col in required_base:
        df[col] = pd.to_numeric(df[col], errors="coerce")

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
        target=target_name,
        group_ids=["group_id"],
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
    )
    val_loader = validation.to_dataloader(
        train=False,
        batch_size=config["BATCH_SIZE"],
        num_workers=config["NUM_WORKERS"],
    )
    test_loader = test.to_dataloader(
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
        training_dataset,
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
