from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2] / "matlab_model_x650_v03" / "results"
OUT = Path(__file__).resolve().parent
SEEDS = list(range(4301, 4321))
AGENTS = [11, 22, 33]
VARIANTS = ["fallback", "alarm63", "early_fallback", "raw_p90", "calibrated_p90"]
METRICS = ["peak_temperature_C", "fan_energy_Wh", "hydrogen_g", "final_SOC", "unmet_energy_kJ"]


def main():
    source = ROOT / "reviewer_alarm_ablation_v01" / "episodes_v01.csv"
    df = pd.read_csv(source)
    expected = pd.MultiIndex.from_product(
        [SEEDS, AGENTS, VARIANTS], names=["scenario_seed", "agent_seed", "variant"]
    )
    actual = pd.MultiIndex.from_frame(df[["scenario_seed", "agent_seed", "variant"]])
    if len(df) != 300 or not expected.difference(actual).empty or actual.has_duplicates:
        raise ValueError("Incomplete or duplicated paired ablation")
    if (df.peak_temperature_C > 70.00001).any():
        raise ValueError("Unexpected out-of-domain episode")

    summary = df.groupby("variant").agg(
        episodes=("scenario_seed", "size"),
        mean_peak_C=("peak_temperature_C", "mean"),
        mean_fan_Wh=("fan_energy_Wh", "mean"),
        mean_hydrogen_g=("hydrogen_g", "mean"),
        mean_final_SOC=("final_SOC", "mean"),
        mean_unmet_kJ=("unmet_energy_kJ", "mean"),
        violated_episodes=("violation_steps", lambda x: int((x > 0).sum())),
        violation_steps=("violation_steps", "sum"),
        mean_predictive_overrides=("predictive_override_steps", "mean"),
        mean_reactive_overrides=("reactive_override_steps", "mean"),
    ).reset_index()
    summary.to_csv(OUT / "Reviewer_alarm_ablation_summary.csv", index=False)

    cluster = df.groupby(["scenario_seed", "variant"])[METRICS].mean()
    rng = np.random.default_rng(20261103)
    draws = rng.integers(0, len(SEEDS), size=(10000, len(SEEDS)))
    contrasts = []
    for comparator in VARIANTS[:-1]:
        for metric in METRICS:
            matrix = cluster[metric].unstack("variant").loc[SEEDS]
            difference = (matrix["calibrated_p90"] - matrix[comparator]).to_numpy()
            resamples = difference[draws].mean(axis=1)
            low, high = np.quantile(resamples, [0.025, 0.975])
            contrasts.append(
                {
                    "contrast": f"calibrated_p90-minus-{comparator}",
                    "metric": metric,
                    "n_scenario_clusters": 20,
                    "n_agent_seeds_per_cluster": 3,
                    "mean_difference": difference.mean(),
                    "bootstrap_low_95": low,
                    "bootstrap_high_95": high,
                }
            )
    paired = pd.DataFrame(contrasts)
    paired.to_csv(OUT / "Reviewer_alarm_ablation_paired_bootstrap.csv", index=False)
    print(summary.round(4).to_string(index=False))
    print(paired[paired.metric.isin(["peak_temperature_C", "fan_energy_Wh"])].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
