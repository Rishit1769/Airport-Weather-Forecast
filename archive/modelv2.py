import pandas as pd
import numpy as np
import torch

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, r2_score

# -------------------------------
# LOAD DATA
# -------------------------------

df = pd.read_csv("your_file.csv")

df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime')

# -------------------------------
# CREATE TIME INDEX
# -------------------------------

df['time_idx'] = (df['datetime'] - df['datetime'].min()).dt.total_seconds() // 1800
df['time_idx'] = df['time_idx'].astype(int)

# Single time series
df['group'] = 0

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------

df['hour'] = df['datetime'].dt.hour
df['month'] = df['datetime'].dt.month
df['dew_diff'] = df['temp'] - df['dew_point']

# -------------------------------
# PARAMETERS
# -------------------------------

max_encoder_length = 24   # past 12 hours
max_prediction_length = 12  # next 6 hours

# -------------------------------
# DATASET
# -------------------------------

training_cutoff = df["time_idx"].max() - max_prediction_length

training = TimeSeriesDataSet(
    df[df.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="visibility",
    group_ids=["group"],

    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,

    time_varying_known_reals=[
        "time_idx", "hour", "month"
    ],

    time_varying_unknown_reals=[
        "visibility",
        "wind_speed",
        "temp",
        "dew_point",
        "humidity",
        "pressure",
        "dew_diff"
    ],

    target_normalizer=GroupNormalizer(groups=["group"]),
)

validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)

train_loader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
val_loader = validation.to_dataloader(train=False, batch_size=64, num_workers=0)

# -------------------------------
# MODEL
# -------------------------------

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=1,
    loss=torch.nn.L1Loss(),
)

# -------------------------------
# TRAINING
# -------------------------------

trainer = Trainer(
    max_epochs=20,
    accelerator="cpu",  # change to "gpu" if available
    callbacks=[EarlyStopping(monitor="val_loss", patience=5)]
)

trainer.fit(tft, train_loader, val_loader)

# -------------------------------
# PREDICTION
# -------------------------------

raw_predictions, x = tft.predict(val_loader, mode="raw", return_x=True)

# Extract predictions
y_pred = raw_predictions.output.numpy().flatten()
y_true = x["decoder_target"].numpy().flatten()

# -------------------------------
# EVALUATION
# -------------------------------

mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print("MAE:", mae)
print("R2:", r2)