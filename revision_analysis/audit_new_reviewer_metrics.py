from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2] / "matlab_model_x650_v03" / "results"
OUT = Path(__file__).resolve().parent


def matched_persistence():
    df = pd.read_csv(
        ROOT / "probability_dataset_v01" / "thermal_trajectories_v01.csv",
        usecols=["scenario_id", "split", "controller", "time_s", "stack_temperature_C"],
    )
    rows = []
    for sid, group in df.groupby("scenario_id", sort=False):
        group = group.sort_values("time_s")
        y = group.stack_temperature_C.to_numpy()
        indices = np.arange(119, len(y) - 20, 10)
        delta = y[indices + 20] - y[indices]
        rows.append(
            {
                "scenario_id": sid,
                "split": group.split.iloc[0],
                "controller": group.controller.iloc[0],
                "n_windows": len(delta),
                "persistence_mae_C": np.mean(np.abs(delta)),
                "fraction_abs_change_lt_0p1_C": np.mean(np.abs(delta) < 0.1),
                "fraction_abs_change_gt_0p5_C": np.mean(np.abs(delta) > 0.5),
                "fraction_start_T_ge_65_C": np.mean(y[indices] >= 65),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "Reviewer_matched_persistence_by_mission.csv", index=False)
    print("MATCHED PERSISTENCE")
    print(result.groupby(["split", "controller"]).mean(numeric_only=True).round(4))


def upper_tail():
    path = ROOT / "input_ablation_replication_v01" / "input_replication_predictions_v01.csv"
    df = pd.read_csv(path)
    rows = []
    for (seed, variant, split), group in df.groupby(["training_seed", "variant", "split"]):
        y = group.actual_temperature_C.to_numpy()
        lo = group.p10_C.to_numpy()
        mid = group.p50_C.to_numpy()
        hi = group.p90_C.to_numpy()
        error = y - mid
        upper_error = y - hi
        largest = max(1, int(round(0.01 * len(error))))
        rows.append(
            {
                "training_seed": seed,
                "variant": variant,
                "split": split,
                "n_windows": len(group),
                "interval_coverage": np.mean((y >= lo) & (y <= hi)),
                "mean_interval_width_C": np.mean(hi - lo),
                "upper_exceedance_rate": np.mean(upper_error > 0),
                "p90_pinball_C": np.mean(np.maximum(0.9 * upper_error, -0.1 * upper_error)),
                "interval_score_80_C": np.mean(
                    (hi - lo) + 10 * np.maximum(lo - y, 0) + 10 * np.maximum(y - hi, 0)
                ),
                "p99_abs_median_error_C": np.quantile(np.abs(error), 0.99),
                "max_abs_median_error_C": np.max(np.abs(error)),
                "largest_1pct_share_squared_error": np.sort(error**2)[-largest:].sum()
                / np.sum(error**2),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "Reviewer_upper_tail_replication_metrics.csv", index=False)
    print("UPPER TAIL")
    print(result.round(4).to_string(index=False))


if __name__ == "__main__":
    matched_persistence()
    upper_tail()
