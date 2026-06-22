# Visibility Specialist

## 1. Overview & Final Metrics

Visibility is the most operationally asymmetric target: errors during clear
conditions are tolerable compared with missing a 150-500 m fog event.

| Metric | Target | Current result |
|---|---:|---:|
| \(R^2\) | \(\ge 0.90\) | **0.9276** |
| RMSE | Lower is better | **316.2796 m** |
| MAE | Lower is better | **207.3757 m** |

> **Warning: source contradicts the historical narrative.** The current
> `mod_visibility.py` is a **single linear-space XGBRegressor**, not an SMoE.
> It has no regime classifier, specialist blend, blend amplifier, or monotonic
> constraints. This document records those ideas in the failed-approach
> graveyard rather than claiming they are deployed.

> **Warning: flatline deletion remains aggressive.** The live run removed
> 152,546 rows, or 87.03%, using `rolling(4).std() < 1.0`. The prompt describes
> this as a past disaster that was fixed, but the current source still performs
> it. The final metric therefore applies only to the retained subset.

## 2. The Problem Space

CSMI visibility ranges from the physical reporting floor near 150 m to the
10,000 m clear-air ceiling. The distribution is discrete and regime-heavy:
METAR observations repeat standard buckets rather than forming a smooth
continuous signal.

Fog and haze require different physics:

- radiation fog favors pre-dawn cooling, low wind, and near-saturation;
- haze reflects aerosol accumulation, humidity growth, and weak mixing;
- monsoon rain can reduce visibility through precipitation and low cloud;
- rapid clearing can occur when wind mixing or solar heating strengthens.

Rare dense-fog rows carry high squared-error cost but represent a tiny fraction
of the annual record.

## 3. Difficulties Faced

### Huber Loss Cowardice

Pseudo-Huber and MAE-style losses reduce the gradient contribution of extreme
residuals. That is desirable for generic outliers but harmful when deep fog is
the event of interest. The model established an artificial prediction floor
and treated 150 m observations as noise.

### Data Deletion Disaster

The flatline rule interprets four consecutive near-identical values as a dead
sensor:

\[
\operatorname{std}(V_{t-3:t}) < 1
\]

METAR visibility is naturally quantized, so valid repeated buckets satisfy
this condition. The current run removes 87.03% of engineered rows. This
produces a cleaner but potentially non-representative evaluation population.

### Rare-Event Imbalance

An 85/15 chronological split placed seasonal dense fog entirely in training.
The current module instead defines three split-only regimes and stratifies:

| Regime | Definition | Train | Test |
|---|---|---:|---:|
| Dense fog | \(V\le500\) m | 53 | 9 |
| Moderate fog/haze | \(500<V\le2000\) m | 3,569 | 630 |
| Clearer | \(V>2000\) m | 15,705 | 2,772 |

### Target-Transformation Distortion

Log, logit, deficit, and inverse-extinction targets improved one region while
distorting linear-space \(R^2\). Reconstruction magnified small transformed
errors into large errors in meters.

## 4. Failed Approaches - The Graveyard

| Approach | Why it failed | Lesson |
|---|---|---|
| Hard hurdle model | Classification mistakes created an MSE cliff at regime boundaries | Do not hard-route a continuous target |
| Soft mixture-of-experts | Added classifier/specialist complexity without surviving into the final source | Prefer the simplest architecture that wins honestly |
| Tweedie deficit target | Zero-inflation and METAR snapping distorted linear residuals | Distributional elegance does not guarantee metric alignment |
| Pseudo-Huber loss | Suppressed gradients from operationally critical fog | Training loss must reflect evaluation and risk |
| Log/square-root/logit targets | Squeezed tails and complicated reconstruction | Transformations can trade clear-day accuracy for fog sensitivity |
| Koschmieder inverse space | Inverse reconstruction distorted linear MSE | Predict in the space used for evaluation |
| Global METAR snapping | Quantization introduced artificial jumps | Post-processing must be calibrated, not assumed |
| Monotonic constraints | Over-regularized splits needed for interacting fog regimes | Local physics may not be globally monotone |
| Prediction floor multiplier | Compressed already-low specialist output | Change mixture influence, not raw predictions |
| Flatline mask at std < 1 | Deleted valid quantized observations | Repetition is not automatically sensor failure |

## 5. Concepts Implemented

### Absolute Linear-Space Regression

\[
y_t = V_t,\qquad L=\sum_i w_i(V_i-\hat{V}_i)^2
\]

Training and evaluation both operate in meters. Predictions are clipped to
the physical interval \([150,10000]\).

### Normalized Inverse-Frequency Weighting

Visibility is divided into ten histogram bins. For observation \(i\):

\[
\tilde{w}_i=\frac{1}{\max(n_{\operatorname{bin}(i)},1)}, \qquad
w_i=\frac{\tilde{w}_i}{\operatorname{mean}(\tilde{w})}
\]

The live mean is exactly 1.0, increasing rare-bin influence without changing
the overall gradient scale.

### Stratified Rare-Event Split

The split label is used only by `train_test_split` and is removed from model
features. Both partitions are sorted by timestamp afterward for plotting.

### Fog-Onset Dynamics

\[
\dot{D}_{1h}=D_t-D_{t-2},\quad
\dot{D}_{2h}=D_t-D_{t-4}
\]

where \(D=T-T_d\). Negative velocity means the air is approaching saturation.

The binary onset signal is:

\[
I(\dot{D}_{1h}<-0.5 \land W_{t-1}<5)
\]

### Persistence Memory

After a lagged observation below 1000 m, fog memory decays as:

\[
M(h)=e^{-h/4}
\]

This encodes the tendency of fog regimes to persist rather than switch
independently every 30 minutes.

```mermaid
flowchart LR
    A[Shared engineered frame] --> B[Visibility physics features]
    B --> C[Flatline exclusion]
    C --> D[Three-regime stratified split]
    D --> E[Inverse-frequency weights]
    E --> F[Single linear-space XGBRegressor]
    F --> G[Clip 150-10000 m]
```

## 6. Feature Engineering Reference

The saved bundle contains 175 active features. Target-specific features are
listed individually; shared features are listed as exact generated families.

| Feature or family | Formula / members | Meteorological purpose |
|---|---|---|
| `vis_lag_1` | \(V_{t-1}\) | Immediate persistence |
| `vis_lag_3h/6h/12h/24h` | shifts 6/12/24/48 | Fog buildup and daily memory |
| `temp_lag_1`, `temp_lag_3` | lagged temperature | Recent cooling |
| `wind_lag_1` | lagged wind speed | Mixing strength |
| `pressure_lag_1` | lagged pressure | Stable-air proxy; calculated but excluded by current drop rule |
| `dew_depression` | \(T_{t-1}-T_{d,t-1}\) | Saturation distance |
| `dew_depression_sq` | \(D^2\) | Nonlinear fog threshold; calculated but excluded because its name contains `vis` only indirectly? It remains active only if selected |
| `dew_depression_velocity_1h/2h` | `diff(2/4)` | Saturation approach rate |
| `vis_velocity_1h/2h` | lagged visibility `diff(2/4)` | Onset/clearance velocity |
| `mixing_layer_proxy` | \(W_{t-1}D_t\) | Wind-dryness mixing; calculated but excluded by current visibility drop filter |
| `fog_onset_signal` | \(I(\dot D_{1h}<-0.5,W<5)\) | Fog trigger |
| `vis_roll_mean_3h/6h` | means of `vis_lag_1`, windows 6/12 | Regime baseline |
| `vis_roll_min_3h/6h` | minima of `vis_lag_1`, windows 6/12 | Recent worst condition |
| `dew_roll_mean_3h`, `dew_roll_min_3h` | rolling dew depression | Saturation persistence |
| `vis_trend_3h` | `vis_lag_3h - vis_lag_12h` | Multi-scale change |
| `fog_deepening` | \(I(\text{trend}<-500)\) | Rapid deterioration; calculated but excluded by current drop rule |
| `hour_sin/cos` | daily cyclic encoding | Radiation-fog timing |
| `nocturnal_fog_score` | 02:00-06:00, \(D<2\), \(W<3\) | Pre-dawn radiation fog |
| `fog_persistence_memory` | \(e^{-h/4}\) | Decaying fog memory |
| `boundary_layer_stability` | `(temp_lag_1-temp_lag_3)/10` fallback | Recent cooling/stability |
| `monsoon_phase` | categorical 0-3 by month | Seasonal weather regime |
| Shared lag families | Lags 1/2/3/6/12 for temperature, wind, gust, visibility, pressure, humidity, direction sin/cos, dew gap, pressure/wind change | Multi-field history |
| Shared rolling families | Means 3/6 and std 3/6/12 | Multi-scale state |
| Shared thermodynamics | dew-point depression, wet bulb, humidity interactions, pressure tendency | Fog/haze physics |
| Shared event flags | rain, fog, haze, cloud cover, high humidity, low wind | Regime context |

> **Note:** The feature filter is name-based and intricate. The saved
> checkpoint bundle, not merely `add_vis_features`, is the authoritative list.
> It currently records all 175 selected names.

## 7. Hyperparameter Reference

| Parameter | Value | Reasoning |
|---|---:|---|
| Objective | `reg:squarederror` | Direct alignment with linear \(R^2\) |
| `n_estimators` | 2000 | Fine rare-event partitioning |
| `learning_rate` | 0.015 | Slow boosting |
| `max_depth` | 8 | Captures interacting fog thresholds |
| `gamma` | 0.5 | Requires meaningful split gain |
| `subsample` | 0.85 | Row regularization |
| `colsample_bytree` | 0.85 | Feature regularization |
| `tree_method` | `hist` | GPU histogram trees |
| `device` | `cuda` | GPU acceleration |
| `random_state` | 42 | Reproducibility |
| Test fraction | 0.15 | Stratified holdout |
| Weight bins | 10 | Inverse-frequency balancing |
| Prediction bounds | 150-10,000 m | Physical/METAR range |

No monotonic constraints are configured in the current model.

## 8. Final Metrics

| Metric | Value | Source |
|---|---:|---|
| \(R^2\) | **0.9276** | Dashboard title |
| RMSE | **316.2796 m** | Current modular run |
| MAE | **207.3757 m** | Current modular run |
| Rows removed | **152,546 (87.03%)** | Current modular run |

The score exceeds the target on the retained stratified sample. Because the
flatline mask removes most rows, it should not be interpreted as full-dataset
performance until that rule is redesigned and revalidated.

## 9. Operational Interpretation

The output is a point estimate in meters, suitable for trend monitoring and
decision support around approach minima, low-visibility procedures, and
runway visual range awareness. It is not an official RVR measurement. Dense
fog remains safety-critical even when aggregate \(R^2\) is high; regime counts,
tail MAE, and event detection should accompany deployment evaluation.
