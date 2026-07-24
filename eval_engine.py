import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

logger = logging.getLogger(__name__)

PLOTS_DIR = Path("artifacts/plots")
DASHBOARD_PATH = PLOTS_DIR / "combined_dashboard.png"


def _series_from_results(results_dict):
    wind = results_dict["wind"]
    wind_v2 = results_dict.get("wind_v2")
    temp_true, temp_pred = results_dict["temp"]
    visibility_true, visibility_pred = results_dict["visibility"]
    panels = [
        (
            "Temperature (C)",
            {
                "index": np.arange(len(temp_true)),
                "y_true": temp_true,
                "y_pred": temp_pred,
                "metrics": {"r2": float(r2_score(temp_true, temp_pred))},
            },
        ),
        ("Pressure (hPa)", results_dict["pressure"]),
        (
            "Visibility (m)",
            {
                "index": np.arange(len(visibility_true)),
                "y_true": visibility_true,
                "y_pred": visibility_pred,
                "metrics": {"r2": float(r2_score(visibility_true, visibility_pred))},
            },
        ),
        (
            _wind_speed_title(wind, wind_v2),
            _wind_speed_panel(wind, wind_v2),
        ),
        (
            _wind_gust_title(wind, wind_v2),
            _wind_gust_panel(wind, wind_v2),
        ),
        (
            _wind_dir_title(wind, wind_v2),
            _wind_dir_panel(wind, wind_v2),
        ),
    ]
    return panels


def _wind_speed_title(wind, wind_v2):
    title = (
        "Wind Speed (kt), "
        f"baseline PICP10-90={wind['wind_speed']['picp_10_90']:.3f}"
    )
    if wind_v2 is not None:
        title += (
            f", MOS PICP10-90={wind_v2['wind_speed']['picp_10_90']:.3f} | "
            f"R2 baseline={wind['wind_speed']['metrics']['r2']:.4f}, "
            f"MOS={wind_v2['wind_speed']['metrics']['r2']:.4f}"
        )
    return title


def _wind_gust_title(wind, wind_v2):
    title = "Wind Gust (kt)"
    if wind_v2 is not None:
        title += (
            f" | R2 baseline={wind['wind_gust']['metrics']['r2']:.4f}, "
            f"MOS={wind_v2['wind_gust']['metrics']['r2']:.4f}"
        )
    return title


def _wind_dir_title(wind, wind_v2):
    title = (
        "Wind Direction (deg), "
        f"baseline circular MAE={wind['wind_dir']['circular_mae_deg']:.2f}"
    )
    if wind_v2 is not None:
        title += f", MOS circular MAE={wind_v2['wind_dir']['circular_mae_deg']:.2f}"
    return title


def _wind_speed_panel(wind, wind_v2):
    panel = {
        "index": wind["index"],
        "y_true": wind["wind_speed"]["y_true"],
        "y_pred": wind["wind_speed"]["y_pred"],
        "metrics": wind["wind_speed"]["metrics"],
        "quantiles": wind["wind_speed"]["quantiles"],
        "comparison": [],
    }
    if wind_v2 is not None:
        panel["comparison"].append(
            {
                "label": "MOS Predicted",
                "index": wind_v2["index"],
                "y": wind_v2["wind_speed"]["y_pred"],
                "color": "tab:orange",
                "alpha": 0.85,
            }
        )
        panel["comparison"].append(
            {
                "label": "MOS NWP Baseline",
                "index": wind_v2["index"],
                "y": wind_v2["wind_speed"]["baseline_pred"],
                "color": "tab:green",
                "alpha": 0.65,
            }
        )
    return panel


def _wind_gust_panel(wind, wind_v2):
    panel = {
        "index": wind["index"],
        "y_true": wind["wind_gust"]["y_true"],
        "y_pred": wind["wind_gust"]["y_pred"],
        "metrics": wind["wind_gust"]["metrics"],
        "comparison": [],
    }
    if wind_v2 is not None:
        panel["comparison"].append(
            {
                "label": "MOS Predicted",
                "index": wind_v2["index"],
                "y": wind_v2["wind_gust"]["y_pred"],
                "color": "tab:orange",
                "alpha": 0.85,
            }
        )
        panel["comparison"].append(
            {
                "label": "MOS NWP Baseline",
                "index": wind_v2["index"],
                "y": wind_v2["wind_gust"]["baseline_pred"],
                "color": "tab:green",
                "alpha": 0.65,
            }
        )
    return panel


def _wind_dir_panel(wind, wind_v2):
    panel = {
        "index": wind["index"],
        "y_true": wind["wind_dir"]["y_true"],
        "y_pred": wind["wind_dir"]["y_pred"],
        "component_metrics": wind["wind_dir"]["component_metrics"],
        "comparison": [],
    }
    if wind_v2 is not None:
        panel["comparison"].append(
            {
                "label": "MOS Predicted",
                "index": wind_v2["index"],
                "y": wind_v2["wind_dir"]["y_pred"],
                "color": "tab:orange",
                "alpha": 0.85,
            }
        )
    return panel


def _r2_for_result(result):
    if "metrics" in result:
        return float(result["metrics"]["r2"])
    component_metrics = result["component_metrics"]
    return float(min(metric["r2"] for metric in component_metrics.values()))


def generate_combined_dashboard(results_dict):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(6, 1, figsize=(18, 24))

    for ax, (title, result) in zip(axes, _series_from_results(results_dict)):
        y_true = np.asarray(result["y_true"], dtype=np.float64)
        y_pred = np.asarray(result["y_pred"], dtype=np.float64)
        index = result["index"]
        ax.plot(index, y_true, label="Actual", linewidth=0.8)
        ax.plot(index, y_pred, label="Baseline Predicted", linewidth=0.8, alpha=0.75)
        if "quantiles" in result:
            lower = np.asarray(result["quantiles"]["q_0.10"], dtype=np.float64)
            upper = np.asarray(result["quantiles"]["q_0.90"], dtype=np.float64)
            ax.fill_between(
                index,
                lower,
                upper,
                alpha=0.18,
                label="Baseline 10-90% interval",
            )
        for comparison in result.get("comparison", []):
            ax.plot(
                comparison["index"],
                np.asarray(comparison["y"], dtype=np.float64),
                label=comparison["label"],
                linewidth=0.8,
                alpha=comparison.get("alpha", 0.8),
                color=comparison.get("color"),
            )
        if "R2 baseline=" not in title and "circular MAE" not in title:
            ax.set_title(f"{title} | R2={_r2_for_result(result):.4f}")
        else:
            ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].tick_params(axis="x", rotation=45)
    axes[-1].set_xlabel("Datetime")
    fig.tight_layout()
    fig.savefig(DASHBOARD_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    for png_path in PLOTS_DIR.glob("*.png"):
        if png_path.resolve() != DASHBOARD_PATH.resolve():
            try:
                png_path.unlink()
            except OSError:
                pass

    for target_name, result in results_dict.items():
        if target_name == "wind":
            logger.info(
                "Wind metrics: speed=%s gust=%s direction_circular_mae=%.3f "
                "picp_10_90=%.4f interval_width=%.4f",
                result["wind_speed"]["metrics"],
                result["wind_gust"]["metrics"],
                result["wind_dir"]["circular_mae_deg"],
                result["wind_speed"]["picp_10_90"],
                result["wind_speed"]["mean_interval_width"],
            )
        elif target_name == "wind_v2":
            logger.info(
                "Wind MOS metrics: speed=%s gust=%s direction_circular_mae=%.3f "
                "picp_10_90=%.4f interval_width=%.4f baseline_speed=%s baseline_gust=%s",
                result["wind_speed"]["metrics"],
                result["wind_gust"]["metrics"],
                result["wind_dir"]["circular_mae_deg"],
                result["wind_speed"]["picp_10_90"],
                result["wind_speed"]["mean_interval_width"],
                result["wind_speed"]["baseline_metrics"],
                result["wind_gust"]["baseline_metrics"],
            )
        elif target_name in {"temp", "visibility"}:
            y_true, y_pred = result
            logger.info("%s R2: %.4f", target_name.title(), r2_score(y_true, y_pred))
        else:
            logger.info("%s metrics: %s", target_name, result["metrics"])
    logger.info("Saved combined dashboard: %s", DASHBOARD_PATH)
    return DASHBOARD_PATH
