"""Conditional P10/P50/P90 thermal forecast baseline using histogram boosting."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_pinball_loss, mean_squared_error

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "results" / "probability_dataset_v01"
SOURCE = DATA_DIR / "thermal_trajectories_v01.csv"
OUT = DATA_DIR / "conditional_quantile_results"
DT = 0.5
HORIZONS_S = (10, 30, 60)
QUANTILES = (0.1, 0.5, 0.9)
FEATURES = [
    "stack_temperature_C", "cathode_air_temperature_C", "propulsion_power_W",
    "bus_load_W", "fc_power_W", "battery_power_W", "SOC", "fan_command",
    "fan_power_W", "stack_heat_W", "stack_to_air_heat_W",
    "air_to_ambient_heat_W", "altitude_m", "ambient_temperature_C",
    "air_density_kgpm3", "mission_phase", "temperature_rate_Cps",
    "power_rate_Wps", "time_fraction",
]


def make_supervised(df: pd.DataFrame, steps: int) -> pd.DataFrame:
    frames = []
    for _, g in df.groupby("scenario_id", sort=False):
        g = g.sort_values("time_s").copy()
        g["temperature_rate_Cps"] = g["stack_temperature_C"].diff().fillna(0) / DT
        g["power_rate_Wps"] = g["propulsion_power_W"].diff().fillna(0) / DT
        g["time_fraction"] = g["time_s"] / g["time_s"].max()
        g["target_C"] = g["stack_temperature_C"].shift(-steps)
        frames.append(g.iloc[:-steps])
    return pd.concat(frames, ignore_index=True)


def metrics(y: np.ndarray, pred: dict[float, np.ndarray]) -> dict:
    raw_crossing = np.mean((pred[0.1] > pred[0.5]) | (pred[0.5] > pred[0.9]))
    ordered = np.sort(np.column_stack([pred[q] for q in QUANTILES]), axis=1)
    p10, p50, p90 = ordered.T
    event = y > 65.0
    alarm = p90 > 65.0
    tp = int(np.sum(event & alarm)); fp = int(np.sum(~event & alarm)); fn = int(np.sum(event & ~alarm))
    return {
        "mae_C": float(mean_absolute_error(y, p50)),
        "rmse_C": float(np.sqrt(mean_squared_error(y, p50))),
        "pinball_p10": float(mean_pinball_loss(y, p10, alpha=0.1)),
        "pinball_p50": float(mean_pinball_loss(y, p50, alpha=0.5)),
        "pinball_p90": float(mean_pinball_loss(y, p90, alpha=0.9)),
        "picp_p10_p90": float(np.mean((y >= p10) & (y <= p90))),
        "mpiw_p10_p90_C": float(np.mean(p90 - p10)),
        "raw_crossing_rate": float(raw_crossing),
        "overtemp_events": int(np.sum(event)),
        "overtemp_recall": float(tp / (tp + fn)) if tp + fn else np.nan,
        "overtemp_precision": float(tp / (tp + fp)) if tp + fp else np.nan,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SOURCE)
    if df.groupby("scenario_id")["split"].nunique().max() != 1:
        raise RuntimeError("Scenario leakage detected")
    results = []
    config = {
        "random_state": 20260910, "max_iter": 160, "learning_rate": 0.06,
        "max_leaf_nodes": 31, "min_samples_leaf": 40, "l2_regularization": 0.2,
    }
    for horizon_s in HORIZONS_S:
        data = make_supervised(df, int(round(horizon_s / DT)))
        train = data[data["split"] == "train"]
        models = {}
        for q in QUANTILES:
            model = HistGradientBoostingRegressor(loss="quantile", quantile=q, **config)
            model.fit(train[FEATURES], train["target_C"])
            models[q] = model
            joblib.dump(model, OUT / f"hgb_quantile_q{int(q*100):02d}_h{horizon_s}s.joblib")
        for split in ("validation", "test", "ood_test"):
            subset = data[data["split"] == split]
            y = subset["target_C"].to_numpy()
            pred = {q: models[q].predict(subset[FEATURES]) for q in QUANTILES}
            results.append({"horizon_s": horizon_s, "split": split,
                            "model": "hist_gradient_boosting_quantile", **metrics(y, pred)})
    table = pd.DataFrame(results)
    table.to_csv(OUT / "conditional_quantile_metrics_v01.csv", index=False)
    with (OUT / "training_config_v01.json").open("w", encoding="utf-8") as f:
        json.dump({"features": FEATURES, "quantiles": QUANTILES,
                   "horizons_s": HORIZONS_S, "model": config}, f, indent=2)
    print(table.to_string(index=False))
    print("CONDITIONAL_QUANTILE_BASELINE_V01_COMPLETE")


if __name__ == "__main__":
    main()
