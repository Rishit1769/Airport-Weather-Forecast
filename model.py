import torch
print(torch.cuda.is_available())
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
import warnings
warnings.filterwarnings("ignore")

# =========================================================
# CONFIG — tweak these to experiment
# =========================================================
CSV_FILE    = "clean_weather_data.csv"
SEQ_LENGTH  = 24     # 24 hours of history (48 × 30-min steps)
PRED_STEP   = 12     # 6 hours ahead      (12 × 30-min steps)
TRAIN_RATIO = 0.8
EPOCHS      = 150
BATCH_SIZE  = 256
PATIENCE    = 20
HIDDEN_SIZE = 256
NUM_LAYERS  = 3
DROPOUT     = 0.25
NUM_HEADS   = 8
LR          = 1e-3

# =========================================================
# 1. LOAD DATA
# =========================================================
df = pd.read_csv(CSV_FILE)
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

print(f"Records loaded : {len(df)}")
print(f"Date range     : {df['datetime'].min()} → {df['datetime'].max()}")
print(f"Temp range     : {df['temp'].min():.1f}°C – {df['temp'].max():.1f}°C")

# =========================================================
# 2. FEATURE ENGINEERING
# =========================================================

# --- Cyclic time (preserves periodicity) ---
df["hour_sin"]      = np.sin(2 * np.pi * df["datetime"].dt.hour / 24)
df["hour_cos"]      = np.cos(2 * np.pi * df["datetime"].dt.hour / 24)
df["month_sin"]     = np.sin(2 * np.pi * df["datetime"].dt.month / 12)
df["month_cos"]     = np.cos(2 * np.pi * df["datetime"].dt.month / 12)
df["doy_sin"]       = np.sin(2 * np.pi * df["datetime"].dt.dayofyear / 365)
df["doy_cos"]       = np.cos(2 * np.pi * df["datetime"].dt.dayofyear / 365)
df["wday_sin"]      = np.sin(2 * np.pi * df["datetime"].dt.weekday / 7)
df["wday_cos"]      = np.cos(2 * np.pi * df["datetime"].dt.weekday / 7)

# --- Wind direction as cyclic (0° and 360° are same!) ---
if "wind_dir" in df.columns:
    df["wdir_sin"] = np.sin(np.deg2rad(df["wind_dir"]))
    df["wdir_cos"] = np.cos(np.deg2rad(df["wind_dir"]))

# --- Rolling stats (multi-window trend context) ---
for w in [3, 6, 12, 24, 48]:
    df[f"temp_r{w}_mean"] = df["temp"].rolling(w, min_periods=1).mean()
    df[f"temp_r{w}_std"]  = df["temp"].rolling(w, min_periods=1).std().fillna(0)

for w in [6, 24]:
    df[f"pres_r{w}"]      = df["pressure"].rolling(w, min_periods=1).mean()
    df[f"wind_r{w}"]      = df["wind_speed"].rolling(w, min_periods=1).mean()

# --- Lag features (explicit look-back) ---
for lag in [1, 2, 3, 6, 12, 24, 48]:
    df[f"temp_lag{lag}"] = df["temp"].shift(lag)

# --- Tendency features (forecaster's edge) ---
df["pres_diff1"]  = df["pressure"].diff(1)
df["pres_diff6"]  = df["pressure"].diff(6)
df["pres_diff12"] = df["pressure"].diff(12)
df["temp_diff1"]  = df["temp"].diff(1)
df["temp_diff6"]  = df["temp"].diff(6)

# --- Visibility log-transform (right-skewed) ---
df["vis_log"] = np.log1p(df["visibility"])

# =========================================================
# 3. FEATURE LIST
# =========================================================
features = [
    # Core meteorological
    "temp", "wind_speed", "pressure", "vis_log",
    # Cyclic time
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "doy_sin", "doy_cos", "wday_sin", "wday_cos",
    # Rolling means
    "temp_r3_mean", "temp_r6_mean", "temp_r12_mean",
    "temp_r24_mean", "temp_r48_mean",
    # Rolling std
    "temp_r3_std", "temp_r6_std", "temp_r12_std",
    # Pressure & wind rolling
    "pres_r6", "pres_r24", "wind_r6", "wind_r24",
    # Lag features
    "temp_lag1", "temp_lag2", "temp_lag3",
    "temp_lag6", "temp_lag12", "temp_lag24", "temp_lag48",
    # Tendency
    "pres_diff1", "pres_diff6", "pres_diff12",
    "temp_diff1", "temp_diff6",
]

# Add wind_dir cyclic if column exists
if "wind_dir" in df.columns:
    features += ["wdir_sin", "wdir_cos"]

df = df.dropna(subset=features).reset_index(drop=True)
INPUT_SIZE = len(features)

print(f"\nFeatures used  : {INPUT_SIZE}")
print(f"Clean samples  : {len(df)}")

# =========================================================
# 4. SCALE — fit only on train split (no data leakage)
# =========================================================
data      = df[features].values
split_raw = int(len(data) * TRAIN_RATIO)

# feature_range=(-1, 1) works better with LSTM tanh activations
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(data[:split_raw])
data_scaled = scaler.transform(data)

# =========================================================
# 5. SEQUENCE BUILDER
# =========================================================
def make_sequences(arr, seq_len, pred_step):
    X, y = [], []
    for i in range(len(arr) - seq_len - pred_step + 1):
        X.append(arr[i : i + seq_len])
        y.append(arr[i + seq_len + pred_step - 1][0])   # temp = column 0
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X_all, y_all = make_sequences(data_scaled, SEQ_LENGTH, PRED_STEP)

split    = int(len(X_all) * TRAIN_RATIO)
X_train, X_test = X_all[:split], X_all[split:]
y_train, y_test = y_all[:split], y_all[split:]

print(f"Train sequences: {len(X_train)}")
print(f"Test  sequences: {len(X_test)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device         : {device}")
if device.type == "cuda":
    print(f"GPU            : {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True

X_train_t = torch.from_numpy(X_train).to(device)
y_train_t = torch.from_numpy(y_train).to(device)
X_test_t  = torch.from_numpy(X_test).to(device)
y_test_t  = torch.from_numpy(y_test).to(device)

# =========================================================
# 6. MODEL — BiLSTM + Multi-Head Temporal Attention
# =========================================================
class TemporalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        out, _ = self.attn(x, x, x)
        return self.norm(x + out)

class WeatherForecastModel(nn.Module):
    def __init__(self, input_size, hidden_size=256,
                 num_layers=3, dropout=0.25, num_heads=8):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )

        self.lstm = nn.LSTM(
            hidden_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        self.bidir_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        self.attn = TemporalAttention(hidden_size, num_heads=num_heads, dropout=0.1)

        self.pool_norm = nn.LayerNorm(hidden_size * 2)
        self.pool_drop = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x, _ = self.lstm(x)
        x = self.bidir_proj(x)
        x = self.attn(x)
        avg = x.mean(dim=1)
        mx  = x.max(dim=1).values
        out = torch.cat([avg, mx], dim=1)
        out = self.pool_norm(out)
        out = self.pool_drop(out)
        return self.head(out)

model = WeatherForecastModel(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    num_heads=NUM_HEADS,
).to(device)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model params   : {n_params:,}")

# =========================================================
# 7. TRAINING SETUP
# =========================================================
criterion = nn.HuberLoss(delta=1.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

def lr_lambda(epoch):
    warmup = 5
    if epoch < warmup:
        return (epoch + 1) / warmup
    prog = (epoch - warmup) / max(1, EPOCHS - warmup)
    return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * prog))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# =========================================================
# 8. TRAIN LOOP WITH EARLY STOPPING
# =========================================================
train_losses, val_losses = [], []
best_val_loss  = float("inf")
best_state     = None
patience_count = 0
n_train        = len(X_train_t)

print("\nTraining...\n" + "─" * 60)

for epoch in range(EPOCHS):
    model.train()
    perm       = torch.randperm(n_train, device=device)
    epoch_loss = 0.0
    n_batches  = 0

    for i in range(0, n_train, BATCH_SIZE):
        idx = perm[i : i + BATCH_SIZE]
        xb  = X_train_t[idx]
        yb  = y_train_t[idx].unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        n_batches  += 1

    scheduler.step()

    model.eval()
    with torch.no_grad():
        val_loss = criterion(
            model(X_test_t), y_test_t.unsqueeze(1)
        ).item()

    avg_train = epoch_loss / n_batches
    train_losses.append(avg_train)
    val_losses.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss  = val_loss
        best_state     = {k: v.clone() for k, v in model.state_dict().items()}
        patience_count = 0
        marker = " ✓ best"
    else:
        patience_count += 1
        marker = ""

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}/{EPOCHS}  "
              f"train={avg_train:.5f}  val={val_loss:.5f}  "
              f"lr={scheduler.get_last_lr()[0]:.6f}{marker}")

    if patience_count >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch + 1}")
        break

model.load_state_dict(best_state)
print(f"\nBest val loss  : {best_val_loss:.6f}")

# =========================================================
# 9. PREDICT & INVERSE TRANSFORM
# =========================================================
model.eval()
with torch.no_grad():
    raw_preds = model(X_test_t).cpu().numpy().flatten()
raw_true  = y_test_t.cpu().numpy()

def inv_temp(vals):
    dummy = np.zeros((len(vals), INPUT_SIZE))
    dummy[:, 0] = vals
    return scaler.inverse_transform(dummy)[:, 0]

preds_actual = inv_temp(raw_preds)
y_actual     = inv_temp(raw_true)

# =========================================================
# 10. EVALUATE
# =========================================================
mae  = mean_absolute_error(y_actual, preds_actual)
rmse = math.sqrt(mean_squared_error(y_actual, preds_actual))
mape = np.mean(np.abs((y_actual - preds_actual) / (np.abs(y_actual) + 1e-8))) * 100
w1   = np.mean(np.abs(y_actual - preds_actual) <= 1.0) * 100
w2   = np.mean(np.abs(y_actual - preds_actual) <= 2.0) * 100
bias = np.mean(preds_actual - y_actual)

print("\n" + "═" * 50)
print("       EVALUATION  —  6-Hour Temperature Forecast")
print("═" * 50)
print(f"  MAE          : {mae:.3f} °C")
print(f"  RMSE         : {rmse:.3f} °C")
print(f"  MAPE         : {mape:.2f} %")
print(f"  Bias (mean)  : {bias:+.3f} °C")
print(f"  Within ±1°C  : {w1:.1f}%  of test predictions")
print(f"  Within ±2°C  : {w2:.1f}%  of test predictions")
print(f"  Test samples : {len(y_actual)}")
print("═" * 50)

# =========================================================
# 11. VISUALIZE
# =========================================================
N      = min(400, len(y_actual))
errors = preds_actual - y_actual

fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.35)

ax1 = fig.add_subplot(gs[0, :])
ax1.plot(y_actual[:N], label="Actual", color="#1565C0", lw=1.5, alpha=0.9)
ax1.plot(preds_actual[:N], label="Predicted (6hr ahead)", color="#E65100", lw=1.2, linestyle="--", alpha=0.85)
ax1.fill_between(range(N), y_actual[:N], preds_actual[:N], alpha=0.10, color="#E65100")
ax1.set_title(
    f"Temperature Forecast — Next 6 Hours\n"
    f"MAE={mae:.2f}°C   RMSE={rmse:.2f}°C   "
    f"Within ±1°C: {w1:.0f}%   Within ±2°C: {w2:.0f}%",
    fontsize=12, fontweight="bold"
)
ax1.set_xlabel("Time Step (30-min intervals)")
ax1.set_ylabel("Temperature (°C)")
ax1.legend(loc="upper right")
ax1.grid(alpha=0.3)

ax2 = fig.add_subplot(gs[1, 0])
ax2.bar(range(N), errors[:N], color=["#c0392b" if e > 0 else "#27ae60" for e in errors[:N]], alpha=0.6, width=1.0)
ax2.axhline(0, color="black", lw=0.8)
ax2.axhline(1, color="#c0392b", lw=0.8, linestyle=":", label="+1°C")
ax2.axhline(-1, color="#27ae60", lw=0.8, linestyle=":", label="-1°C")
ax2.set_title("Prediction Error (Predicted − Actual)", fontsize=11, fontweight="bold")
ax2.set_xlabel("Time Step")
ax2.set_ylabel("Error (°C)")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

ax3 = fig.add_subplot(gs[1, 1])
ax3.hist(errors, bins=60, color="#1976D2", alpha=0.75, edgecolor="white")
ax3.axvline(0, color="red", lw=1.5, linestyle="--", label="Zero error")
ax3.axvline(np.mean(errors), color="orange", lw=1.2, linestyle="-.", label=f"Mean={np.mean(errors):+.2f}°C")
ax3.set_title("Error Distribution", fontsize=11, fontweight="bold")
ax3.set_xlabel("Error (°C)")
ax3.set_ylabel("Count")
ax3.legend()
ax3.grid(alpha=0.3)

ax4 = fig.add_subplot(gs[2, 0])
ax4.plot(train_losses, label="Train Loss", color="#2ecc71", lw=1.5)
ax4.plot(val_losses, label="Val Loss", color="#e74c3c", lw=1.5)
best_ep = int(np.argmin(val_losses))
ax4.axvline(best_ep, color="gray", lw=1.0, linestyle=":", label=f"Best epoch {best_ep+1}")
ax4.set_title("Training & Validation Loss (Huber)", fontsize=11, fontweight="bold")
ax4.set_xlabel("Epoch")
ax4.set_ylabel("Loss")
ax4.legend()
ax4.grid(alpha=0.3)

ax5 = fig.add_subplot(gs[2, 1])
ax5.scatter(y_actual, preds_actual, alpha=0.2, s=5, color="#7B1FA2")
mn, mx = y_actual.min(), y_actual.max()
ax5.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect fit")
ax5.set_title("Actual vs Predicted (Scatter)", fontsize=11, fontweight="bold")
ax5.set_xlabel("Actual (°C)")
ax5.set_ylabel("Predicted (°C)")
ax5.legend()
ax5.grid(alpha=0.3)

plt.suptitle("Weather Forecast — BiLSTM + Multi-Head Temporal Attention", fontsize=14, fontweight="bold", y=1.01)
plt.savefig("forecast_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved → forecast_results.png")

# =========================================================
# 12. SAVE MODEL
# =========================================================
torch.save({
    "model_state": model.state_dict(),
    "input_size": INPUT_SIZE,
    "hidden_size": HIDDEN_SIZE,
    "num_layers": NUM_LAYERS,
    "num_heads": NUM_HEADS,
    "dropout": DROPOUT,
    "seq_length": SEQ_LENGTH,
    "pred_step": PRED_STEP,
    "features": features,
    "mae": mae,
    "rmse": rmse,
}, "weather_model_best.pth")

print("Model saved → weather_model_best.pth")