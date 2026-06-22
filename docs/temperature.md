# Temperature Specialist

## 1. Overview & Final Metrics

The temperature specialist forecasts the next 30-minute CSMI temperature by
predicting the change from the latest observation rather than the absolute
temperature directly.

| Metric | Target | Current result |
|---|---:|---:|
| \(R^2\) | \(\ge 0.95\) | **0.9542** |
| RMSE | Lower is better | **0.5746 C** |
| MAE | Lower is better | **0.4187 C** |
| Lag-1 persistence \(R^2\) | Diagnostic baseline | **0.9276** |

The \(R^2\) value is read from `artifacts/plots/combined_dashboard.png`. RMSE,
MAE, and the persistence baseline come from the current modular pipeline run
captured in `artifacts/current_pipeline_run.log`.

> **Warning: stale metrics artifact.** `artifacts/eval_metrics.json` describes
> the archived monolithic pipeline and reports temperature \(R^2=0.8235\).
> It is not consistent with `mod_temperature.py`, the current checkpoint, or
> the current dashboard, so it is not used as a final modular metric.

## 2. The Problem Space

Mumbai's coastal climate compresses the annual temperature range, but the
30-minute signal is not trivial. Slow radiative warming is interrupted by
sea-breeze advection, convective outflow, rain-cooled air, cloud changes, and
rapid pre-dawn stabilization. The model must distinguish ordinary persistence
from an actual change in the thermal field.

At this cadence, a persistence model can score well while remaining
operationally late. A two-step delay is already one hour behind a sea-breeze
onset or convective cooling event.

## 3. Difficulties Faced

### Persistence Trap

For smooth temperature data, minimizing squared error encourages:

\[
\hat{T}_{t+1} \approx T_t
\]

This is statistically attractive because most 30-minute changes are small.
The failure appears during thermal transitions: predictions trail the observed
peak or trough by several samples. The model is accurate when the atmosphere
does nothing and late precisely when operational conditions change.

Adding more absolute-temperature lags deepened this behavior. EMA features
improved context but could not force the estimator to learn the sign and
magnitude of the next change.

### Feature-Level Target Leakage

Early rolling statistics were calculated directly from the current
temperature. A window containing \(T_t\) can allow a tree to algebraically
recover the answer. The current module calculates every temperature rolling
or exponential statistic from `temp_lag_1`, and explicitly rejects
current-temperature-derived shared features.

### Dead-Sensor Flatlines

Four hours of exactly zero accumulated movement are treated as a locked
temperature sensor. Those rows are removed before splitting so synthetic
forward-filled plateaus are not learned or evaluated.

## 4. Failed Approaches - The Graveyard

| Approach | Why it failed | Engineering lesson |
|---|---|---|
| Absolute temperature regression | Collapsed toward \(T_t\) and lagged transitions | High \(R^2\) does not prove a model learned dynamics |
| More lag depth alone | Added memory but strengthened persistence | History requires a target that rewards change |
| Rolling features on current `temp` | Embedded the target inside averages and extrema | All temporal statistics must be causal |
| EMA-only upgrade | Smoothed the state but did not force derivative learning | State memory and target formulation solve different problems |
| Unfiltered shared features | Included present-state thermal interactions | Target-specific feature allowlists are safer than broad drops |

## 5. Concepts Implemented

### Delta Prediction

The training target is:

\[
\Delta T_t = T_t - T_{t-1}
\]

Inference reconstructs absolute temperature:

\[
\hat{T}_t = T_{t-1} + \widehat{\Delta T_t}
\]

Predicting zero is now an explicit "no change" forecast rather than an
implicit copy of the target. Heating rate, cooling rate, humidity history, and
solar phase must explain departures from persistence.

### Thermal Inertia with EMA

For a span \(s\), pandas uses:

\[
\alpha = \frac{2}{s+1}, \qquad
\operatorname{EMA}_t = \alpha x_t + (1-\alpha)\operatorname{EMA}_{t-1}
\]

The module uses lagged temperature with spans 6 and 12:

| Feature | Span | \(\alpha\) | Approximate history |
|---|---:|---:|---:|
| `temp_ema_3h` | 6 | \(2/7 \approx 0.2857\) | 3 hours |
| `temp_ema_6h` | 12 | \(2/13 \approx 0.1538\) | 6 hours |

These features approximate atmospheric thermal memory: recent air-mass state
matters more than equally weighted older samples.

### Phase-Shifted Solar Anchor

\[
S_t = \sin\left(\frac{2\pi(h_t-8.5)}{24}\right)
\]

The `8.5`-hour phase makes the sine maximum at 14:30, representing the lag
between solar forcing and maximum surface-air temperature.

### Coastal Dew-Point Floor

\[
D_t = T_{t-1} - T_{d,t-1}
\]

Small dew-point depression indicates a nearly saturated coastal boundary
layer and limits plausible nocturnal cooling.

```mermaid
flowchart LR
    A[Master weather frame] --> B[Causal temperature features]
    B --> C[Outage mask]
    C --> D[Chronological train/validation/test]
    D --> E[XGBoost predicts delta T]
    E --> F[Add temp_lag_1]
    F --> G[Absolute temperature forecast]
```

## 6. Feature Engineering Reference

The active checkpoint contains 80 features. The table lists every
temperature-specific feature and every inherited feature family.

| Feature or family | Formula / members | Meteorological purpose |
|---|---|---|
| `temp_lag_1/2/3` | \(T_{t-k}\), \(k\in\{1,2,3\}\) | Immediate thermal state |
| `temp_diff_1h` | `temp_lag_1 - temp_lag_3` | One-hour heating/cooling velocity |
| `temp_lag_24h`, `temp_lag_48h` | shifts 48 and 96 | Diurnal recurrence |
| `hour_sin`, `hour_cos` | \(\sin/\cos(2\pi h/24)\) | Circular time-of-day encoding |
| `month_sin`, `month_cos` | \(\sin/\cos(2\pi m/12)\) | Seasonal cycle |
| `solar_thermal_peak` | \(\sin(2\pi(h-8.5)/24)\) | Delayed afternoon heat peak |
| `temp_ema_3h`, `temp_ema_6h` | EMA of `temp_lag_1`, spans 6/12 | Thermal inertia |
| `temp_roll_mean_3h` | mean of `temp_lag_1`, window 6 | Recent equilibrium |
| `temp_roll_max_6h`, `temp_roll_min_6h` | extrema of `temp_lag_1`, window 12 | Recent thermal envelope |
| `dew_point_lag_1` | \(T_{d,t-1}\) | Moisture boundary |
| `temp_dew_depression` | \(T_{t-1}-T_{d,t-1}\) | Saturation proximity |
| Shared lag families | `temp`, `wind_speed`, `wind_gust`, `visibility`, `pressure`, `humidity`, wind-direction sin/cos, `temp_dew_diff`, `pressure_change`, `wind_speed_change` at lags 1,2,3,6,12 | Cross-variable causal history |
| Dew-point lags | `dew_point_depression_lag_1/2/3/6` | Moisture trend |
| Wet-bulb lags | `wet_bulb_lag_1/2` | Combined heat/moisture state |
| Visibility-wind interactions | `wind_speed_x_visibility_lag_1/3/6` | Advective and cloud/rain context |
| `dew_gap_lag_3` | lagged absolute temperature/dew-point gap | Saturation memory |

Current-temperature interactions such as raw `wet_bulb`, raw
`dew_point_depression`, `temp_humidity`, and unsafe `temp_rolling_*` shared
features are deliberately excluded.

## 7. Hyperparameter Reference

| Parameter | Value | Reasoning |
|---|---:|---|
| Objective | `reg:squarederror` | Direct optimization of delta residuals |
| `n_estimators` | 2000 | Capacity for fine thermal interactions |
| `learning_rate` | 0.015 | Slow, precise boosting |
| `max_depth` | 7 | Captures nonlinear solar/moisture interactions |
| `subsample` | 0.85 | Row regularization |
| `colsample_bytree` | 0.85 | Feature regularization |
| `gamma` | 0.0 | No extra split penalty |
| `reg_alpha` | 0.0 | No L1 penalty |
| `reg_lambda` | 1.0 | Standard L2 smoothing |
| `early_stopping_rounds` | 50 | Stops against chronological validation |
| `tree_method` | `hist` | GPU-compatible histogram trees |
| `device` | `cuda` | GPU training |
| `random_state` | 42 | Reproducibility |
| Split | 85/15 chronological; final 10% of train used for validation | Preserves temporal order |

## 8. Final Metrics

| Metric | Value | Source |
|---|---:|---|
| \(R^2\) | **0.9542** | Dashboard title |
| RMSE | **0.5746 C** | Current modular run |
| MAE | **0.4187 C** | Current modular run |
| Persistence \(R^2\) | **0.9276** | Current modular run |

The improvement over persistence is the key result: the model adds predictive
value beyond simply repeating the previous reading.

## 9. Operational Interpretation

Sub-degree MAE supports short-horizon runway and ramp temperature awareness,
aircraft performance calculations, and detection of rapid cooling associated
with sea-breeze or convective transitions. Mumbai does not normally require
de-icing support; the more relevant local use is anticipating heat stress,
density-altitude effects, and sudden boundary-layer changes.
