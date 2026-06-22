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
    temp_true, temp_pred = results_dict["temp"]
    visibility_true, visibility_pred = results_dict["visibility"]
    return [
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
            "Wind Speed (kt)",
            {"index": wind["index"], **wind["wind_speed"]},
        ),
        (
            "Wind Gust (kt)",
            {"index": wind["index"], **wind["wind_gust"]},
        ),
        (
            f"Wind Direction (deg), circular MAE={wind['wind_dir']['circular_mae_deg']:.2f}",
            {"index": wind["index"], **wind["wind_dir"]},
        ),
    ]


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
        ax.plot(index, y_pred, label="Predicted", linewidth=0.8, alpha=0.8)
        ax.set_title(f"{title} | R2={_r2_for_result(result):.4f}")
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
                "Wind metrics: speed=%s gust=%s direction_circular_mae=%.3f",
                result["wind_speed"]["metrics"],
                result["wind_gust"]["metrics"],
                result["wind_dir"]["circular_mae_deg"],
            )
        elif target_name in {"temp", "visibility"}:
            y_true, y_pred = result
            logger.info("%s R2: %.4f", target_name.title(), r2_score(y_true, y_pred))
        else:
            logger.info("%s metrics: %s", target_name, result["metrics"])
    logger.info("Saved combined dashboard: %s", DASHBOARD_PATH)
    return DASHBOARD_PATH
