from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
PAIRS = [
    (
        "development",
        PROJECT / "matlab_model_x650_v03/results/input_ablation_replication_v01/input_replication_predictions_v01.csv",
        HERE / "corrected_replication_predictions/input_replication_predictions_v01.csv",
    ),
    (
        "prospective_holdout",
        PROJECT / "matlab_model_x650_v03/results/forecast_holdout_20260912/input_replication_predictions_v01.csv",
        HERE / "corrected_holdout_predictions/input_replication_predictions_v01.csv",
    ),
]


def main():
    rows = []
    changed_alarms = []
    for cohort, old_path, new_path in PAIRS:
        old = pd.read_csv(old_path)
        new = pd.read_csv(new_path)
        keys = ["training_seed", "variant", "split", "scenario_id", "controller"]
        assert len(old) == len(new) and old[keys].equals(new[keys])
        assert np.array_equal(old.actual_temperature_C, new.actual_temperature_C)
        for (seed, variant, split), indices in new.groupby(
            ["training_seed", "variant", "split"]
        ).indices.items():
            a = old.iloc[indices]
            b = new.iloc[indices]
            changed = (a.p90_C.to_numpy() >= 63) != (b.p90_C.to_numpy() >= 63)
            y = b.actual_temperature_C.to_numpy()
            rows.append(
                {
                    "cohort": cohort,
                    "training_seed": seed,
                    "variant": variant,
                    "split": split,
                    "n_windows": len(indices),
                    "mean_abs_p90_shift_C": np.mean(np.abs(a.p90_C - b.p90_C)),
                    "max_abs_p90_shift_C": np.max(np.abs(a.p90_C - b.p90_C)),
                    "original_mae_C": np.mean(np.abs(y - a.p50_C)),
                    "corrected_input_mae_C": np.mean(np.abs(y - b.p50_C)),
                    "changed_63C_alarms": int(changed.sum()),
                    "changed_true_63C_events": int(((y >= 63) & changed).sum()),
                }
            )
            if changed.any():
                c = b.iloc[np.flatnonzero(changed)].copy()
                c["cohort"] = cohort
                c["old_p90_C"] = a.iloc[np.flatnonzero(changed)].p90_C.to_numpy()
                changed_alarms.append(c)
    pd.DataFrame(rows).to_csv(HERE / "R2_corrected_predictor_input_impact.csv", index=False)
    if changed_alarms:
        pd.concat(changed_alarms).to_csv(HERE / "R2_corrected_input_changed_alarm_windows.csv", index=False)
    print(pd.DataFrame(rows).groupby(["cohort", "variant", "split"])[
        ["mean_abs_p90_shift_C", "original_mae_C", "corrected_input_mae_C",
         "changed_63C_alarms", "changed_true_63C_events"]
    ].agg({"mean_abs_p90_shift_C": "mean", "original_mae_C": "mean",
           "corrected_input_mae_C": "mean", "changed_63C_alarms": "sum",
           "changed_true_63C_events": "sum"}).round(5))


if __name__ == "__main__":
    main()
