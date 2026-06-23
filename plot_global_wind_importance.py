from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from xgboost.core import XGBoostError

from data_pipeline import get_engineered_data
from mod_wind import _apply_anemometer_mask, add_wind_features


OUTPUT_PLOT = Path("wind_global_parameters_ranking.png")
OUTPUT_TABLE = Path("wind_global_parameters_ranking.csv")


def load_master_data() -> pd.DataFrame:
    """Project-wide shared loader for the fully engineered weather frame."""
    return get_engineered_data()


def _build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    y_target = np.log1p(df["wind_speed"])

    # 3. STRICT LEAK SCRUBBING
    # Standard base drops
    drop_cols = [
        "wind_speed",
        "wind_gust",
        "wind_direction",
        "wind_dir",
        "u_wind",
        "v_wind",
        "wind_dir_sin",
        "wind_dir_cos",
        "datetime",
    ]

    # THE EXPLICIT LEAK LIST: Drop the cheats revealed in the previous graph
    explicit_leaks = [
        "low_wind_flag",
        "wind_speed_change",
        "wind_speed_change_rolling_mean_3",
        "wind_gust_rolling_mean_3",
        "wind_gust_rolling_mean_6",
        "wind_speed_rolling_mean_3",
        "ke",
    ]
    drop_cols.extend(explicit_leaks)

    # Dynamic Catch-All for any other direct target derivatives.
    # Drops 'target' columns and 'ke_' prefix columns (like ke_volatility_1h).
    drop_cols.extend(
        [
            column
            for column in df.columns
            if "target" in column.lower() or column.startswith("ke_")
        ]
    )

    features = [
        column
        for column in df.columns
        if column not in drop_cols and pd.api.types.is_numeric_dtype(df[column])
    ]
    X = df[features].replace([np.inf, -np.inf], np.nan)
    valid_mask = X.notna().all(axis=1) & y_target.notna()
    X = X.loc[valid_mask].copy()
    y_target = y_target.loc[valid_mask].copy()

    print(f"      -> Ingesting {len(features)} total parameters for importance ranking...")
    print(f"      -> Retained {len(X):,} valid rows after filtering non-finite values.")
    return X, y_target, features


def _fit_model(X: pd.DataFrame, y_target: pd.Series) -> XGBRegressor:
    print("      -> Fitting Global Feature Diagnostic Model...")
    model_params = {
        "n_estimators": 2500,
        "learning_rate": 0.015,
        "max_depth": 8,
        "gamma": 0.1,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "tree_method": "hist",
        "device": "cuda",
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": -1,
    }

    model = XGBRegressor(**model_params)
    try:
        model.fit(X, y_target, verbose=False)
    except XGBoostError as exc:
        if "cuda" not in str(exc).lower():
            raise
        print("      -> CUDA was unavailable; retrying on CPU with the same diagnostic configuration.")
        model_params["device"] = "cpu"
        model = XGBRegressor(**model_params)
        model.fit(X, y_target, verbose=False)
    return model


def extract_global_importance() -> pd.DataFrame:
    print("Loading master weather frame...")
    df_master = load_master_data()

    print("Applying wind-specific feature engineering...")
    df = add_wind_features(df_master)

    print("Applying anemometer quality mask...")
    df = _apply_anemometer_mask(df)

    X, y_target, features = _build_feature_matrix(df)
    model = _fit_model(X, y_target)

    importances = model.feature_importances_
    feat_df = (
        pd.DataFrame({"Parameter": features, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    feat_df.to_csv(OUTPUT_TABLE, index=False)
    print(f"      -> Full importance table saved as {OUTPUT_TABLE}")

    top_features = feat_df.head(20).sort_values("Importance", ascending=True)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(14, 10))
    bars = ax.barh(
        top_features["Parameter"],
        top_features["Importance"],
        color="#00BFFF",
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_title(
        "Global Feature Importance: Wind Speed Determinants",
        fontsize=18,
        pad=20,
        fontweight="bold",
    )
    ax.set_xlabel("Relative Importance (F-Score / Gain)", fontsize=14)
    ax.set_ylabel("Atmospheric Parameters", fontsize=14)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    max_importance = float(top_features["Importance"].max()) if not top_features.empty else 0.0
    offset = max(max_importance * 0.015, 0.002)
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.4f}",
            va="center",
            ha="left",
            fontsize=11,
            color="white",
        )

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches="tight")
    print(f"      -> Feature Importance Graph successfully saved as {OUTPUT_PLOT}")
    plt.show()
    plt.close(fig)
    return feat_df


if __name__ == "__main__":
    extract_global_importance()
