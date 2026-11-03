"""Audit scenario splits and fit persistence/ridge probabilistic temperature baselines."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "results" / "probability_dataset_v01"
TRAJECTORY_FILE = DATA_DIR / "thermal_trajectories_v01.csv"
OUT_DIR = DATA_DIR / "baseline_results"
HORIZON_SECONDS = (10, 30, 60)
FEATURES = [
    "stack_temperature_C",
    "cathode_air_temperature_C",
    "propulsion_power_W",
    "fc_power_W",
    "SOC",
    "fan_command",
    "stack_heat_W",
    "altitude_m",
    "ambient_temperature_C",
    "mission_phase",
]


def audit_splits(df: pd.DataFrame) -> dict:
    scenario_splits = df.groupby("scenario_id")["split"].nunique()
    if int(scenario_splits.max()) != 1:
        raise RuntimeError("Leakage: at least one scenario occurs in multiple splits")
    split_sets = {
        name: set(group["scenario_id"].unique()) for name, group in df.groupby("split")
    }
    names = list(split_sets)
    overlaps = {
        f"{names[i]}__{names[j]}": len(split_sets[names[i]] & split_sets[names[j]])
        for i in range(len(names))
        for j in range(i + 1, len(names))
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Leakage across split scenario IDs: {overlaps}")
    return {
        "rows": int(len(df)),
        "scenarios_by_split": {k: len(v) for k, v in split_sets.items()},
        "scenario_overlap_counts": overlaps,
    }


def supervised_rows(df: pd.DataFrame, horizon_steps: int) -> pd.DataFrame:
    frames = []
    for _, trajectory in df.groupby("scenario_id", sort=False):
        trajectory = trajectory.sort_values("time_s").copy()
        trajectory["temperature_target_C"] = trajectory["stack_temperature_C"].shift(
            -horizon_steps
        )
        trajectory["temperature_rate_Cps"] = trajectory["stack_temperature_C"].diff().fillna(0) / 0.5
        trajectory["power_rate_Wps"] = trajectory["propulsion_power_W"].diff().fillna(0) / 0.5
        frames.append(trajectory.iloc[:-horizon_steps])
    return pd.concat(frames, ignore_index=True)


def interval_metrics(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict:
    return {
        "picp_p10_p90": float(np.mean((y >= lower) & (y <= upper))),
        "mpiw_p10_p90_C": float(np.mean(upper - lower)),
        "crossing_rate": float(np.mean(lower > upper)),
    }


def evaluate(y: np.ndarray, p50: np.ndarray, p10: np.ndarray, p90: np.ndarray) -> dict:
    result = {
        "mae_C": float(mean_absolute_error(y, p50)),
        "rmse_C": float(np.sqrt(mean_squared_error(y, p50))),
    }
    result.update(interval_metrics(y, p10, p90))
    return result


def main() -> None:
    if not TRAJECTORY_FILE.exists():
        raise FileNotFoundError(
            f"Missing {TRAJECTORY_FILE}. Run generate_probability_dataset_v01 in MATLAB first."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(TRAJECTORY_FILE)
    audit = audit_splits(df)
    dt_values = df.groupby("scenario_id")["time_s"].diff().dropna().round(9).unique()
    if len(dt_values) != 1 or not np.isclose(dt_values[0], 0.5):
        raise RuntimeError(f"Unexpected or inconsistent time step: {dt_values}")

    records = []
    extra_features = ["temperature_rate_Cps", "power_rate_Wps"]
    for horizon_s in HORIZON_SECONDS:
        horizon_steps = int(round(horizon_s / 0.5))
        data = supervised_rows(df, horizon_steps)
        train = data[data["split"] == "train"]
        validation = data[data["split"] == "validation"]
        if train.empty or validation.empty:
            raise RuntimeError("Training and validation splits must be non-empty")

        x_train = train[FEATURES + extra_features]
        y_train = train["temperature_target_C"].to_numpy()
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(x_train, y_train)

        validation_prediction = model.predict(validation[FEATURES + extra_features])
        validation_residual = validation["temperature_target_C"].to_numpy() - validation_prediction
        residual_q10, residual_q90 = np.quantile(validation_residual, [0.1, 0.9])

        for split in ("validation", "test", "ood_test"):
            subset = data[data["split"] == split]
            if subset.empty:
                continue
            y = subset["temperature_target_C"].to_numpy()

            persistence = subset["stack_temperature_C"].to_numpy()
            persistence_residual = validation["temperature_target_C"].to_numpy() - validation[
                "stack_temperature_C"
            ].to_numpy()
            pq10, pq90 = np.quantile(persistence_residual, [0.1, 0.9])
            metrics = evaluate(y, persistence, persistence + pq10, persistence + pq90)
            records.append({"horizon_s": horizon_s, "split": split, "model": "persistence", **metrics})

            ridge = model.predict(subset[FEATURES + extra_features])
            metrics = evaluate(y, ridge, ridge + residual_q10, ridge + residual_q90)
            records.append({"horizon_s": horizon_s, "split": split, "model": "ridge_residual_quantiles", **metrics})

    results = pd.DataFrame(records)
    results.to_csv(OUT_DIR / "probability_baseline_metrics_v01.csv", index=False)
    with (OUT_DIR / "dataset_audit_v01.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
    print(results.to_string(index=False))
    print("PROBABILITY_BASELINES_V01_COMPLETE")


if __name__ == "__main__":
    main()
