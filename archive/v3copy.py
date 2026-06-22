import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pytorch_forecasting.data import GroupNormalizer

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.metrics import RMSE
except ImportError as exc:
    raise ImportError(
        "Missing training dependencies. Install from requirements.txt before running v3copy.py."
    ) from exc


CONFIG = {
    "CSV_FILE": "clean_weather_data.csv",
    "SEED": 42,
    "ENCODER_LENGTH": 48,
    "PREDICTION_LENGTH": 6,
    "TRAIN_RATIO": 0.70,
    "VAL_RATIO": 0.15,
    "BATCH_SIZE": 128,
    "MAX_EPOCHS": 100,
    "PATIENCE": 10,
    "LEARNING_RATE": 3e-4,
    "HIDDEN_SIZE": 64,
    "ATTN_HEADS": 4,
    "DROPOUT": 0.2,
    "HIDDEN_CONT_SIZE": 32,
    "NUM_WORKERS": 0,
    "USE_CUDA_IF_AVAILABLE": True,
}

TARGETS = ["temp"]
TIME_FEATURES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos", "is_weekend"]
OBSERVED_FEATURES = [
    "humidity",
    "wind_u",
    "wind_v",
    "temp_trend",
    "dewpoint_approx",

    "is_rain",
    "is_fog",
    "is_haze",

    "dewpoint_spread",

    "pressure_trend_6",
    "pressure_trend_24",

    "vis_trend_6",

    "humidity_trend_6",

    "wind_acceleration",

    "is_monsoon",

    "rain_last_3h",
    "rain_last_6h",

    "fog_last_3h",

    "haze_last_3h"
]


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

    # Weather event indicators
    for col in ["is_rain", "is_fog", "is_haze"]:
        if col not in df.columns:
            df[col] = 0.0

    # Monsoon flag
    month = df["datetime"].dt.month
    df["is_monsoon"] = month.isin([6, 7, 8, 9]).astype(float)

    # Dew point spread
    if "dew_point" in df.columns:
        df["dewpoint_spread"] = df["temp"] - df["dew_point"]
    else:
        df["dewpoint_spread"] = df["temp"] - df["dewpoint_approx"]

    # Pressure trends
    df["pressure_trend_6"] = df["pressure"] - df["pressure"].shift(6)
    df["pressure_trend_24"] = df["pressure"] - df["pressure"].shift(24)

    # Visibility trend
    df["vis_trend_6"] = df["visibility"] - df["visibility"].shift(6)

    # Humidity trend
    df["humidity_trend_6"] = df["humidity"] - df["humidity"].shift(6)

    # Wind acceleration
    df["wind_acceleration"] = df["wind_speed"] - df["wind_speed"].shift(1)

    # Event persistence
    df["rain_last_3h"] = df["is_rain"].rolling(6, min_periods=1).sum()
    df["rain_last_6h"] = df["is_rain"].rolling(12, min_periods=1).sum()
    df["fog_last_3h"] = df["is_fog"].rolling(6, min_periods=1).sum()
    df["haze_last_3h"] = df["is_haze"].rolling(6, min_periods=1).sum()

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
        target_normalizer=GroupNormalizer(
            groups=["group_id"]
        ),
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
        callbacks=[early_stop, checkpoint],
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

    result = model.predict(
        test_loader,
        mode="prediction",
        return_x=True
    )

    y_pred = result.output.detach().cpu().numpy().reshape(-1)

    y_true = (
        result.x["decoder_target"]
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )
    print("y_pred shape:", y_pred.shape)
    print("y_true shape:", y_true.shape)

    print("\nFirst 20 predictions:")
    print(y_pred[:20])

    print("\nFirst 20 actuals:")
    print(y_true[:20])

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    print("\nPrediction Mean:", np.mean(y_pred))
    print("Prediction Std :", np.std(y_pred))

    print("\nActual Mean:", np.mean(y_true))
    print("Actual Std :", np.std(y_true))

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


def main():
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision("high")
    set_seed(CONFIG["SEED"])

    csv_path = Path(CONFIG["CSV_FILE"])
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df = load_data(str(csv_path))
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