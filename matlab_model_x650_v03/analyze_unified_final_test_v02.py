"""Locked final-test integrity checks and paired bootstrap confidence intervals."""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "predictive_safety_v02"
SOURCE = OUT / "unified_final_test_episodes_v02.csv"
METRICS = ["peak_temperature_C", "fan_energy_Wh", "hydrogen_g", "final_SOC", "unmet_energy_kJ"]
N_BOOT = 10000
RNG = np.random.default_rng(20260911)


def bootstrap_mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    samples = RNG.choice(values, size=(N_BOOT, len(values)), replace=True).mean(axis=1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return float(values.mean()), float(lo), float(hi)


def paired_rows(df: pd.DataFrame, left: str, right: str, scenario_ids: set[int], scope: str):
    a = df[(df.policy == left) & df.scenario_seed.isin(scenario_ids)].set_index("scenario_seed")
    b = df[(df.policy == right) & df.scenario_seed.isin(scenario_ids)].set_index("scenario_seed")
    if set(a.index) != set(b.index):
        raise RuntimeError(f"Unpaired scenarios for {left} vs {right}")
    rows = []
    for metric in METRICS:
        delta = b.loc[a.index, metric].to_numpy() - a[metric].to_numpy()
        mean, lo, hi = bootstrap_mean_ci(delta)
        rows.append({"scope": scope, "left": left, "right": right, "metric": metric,
                     "n_scenarios": len(delta), "mean_delta_right_minus_left": mean,
                     "bootstrap_ci95_low": lo, "bootstrap_ci95_high": hi})
    return rows


def main() -> None:
    df = pd.read_csv(SOURCE)
    expected = {"safe_pi", "mpc", "sac_seed11_legacy", "sac_seed11_p90",
                "sac_seed22_legacy", "sac_seed22_p90", "sac_seed33_legacy", "sac_seed33_p90"}
    counts = df.groupby("policy").scenario_seed.nunique().to_dict()
    if set(counts) != expected or any(v != 20 for v in counts.values()) or len(df) != 160:
        raise RuntimeError(f"Incomplete final test: rows={len(df)}, counts={counts}")
    baseline = df[df.policy.isin(["safe_pi", "mpc"])].pivot(
        index="scenario_seed", columns="policy", values="unmet_energy_kJ"
    )
    feasible_ids = set(baseline.index[(baseline.safe_pi < 0.1) & (baseline.mpc < 0.1)])
    all_ids = set(df.scenario_seed.unique())
    rows = []
    for scope, ids in [("all", all_ids), ("architecture_feasible", feasible_ids)]:
        for seed in (11, 22, 33):
            rows += paired_rows(df, f"sac_seed{seed}_legacy", f"sac_seed{seed}_p90", ids, scope)
            rows += paired_rows(df, "mpc", f"sac_seed{seed}_p90", ids, scope)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "unified_final_test_bootstrap_v02.csv", index=False)
    audit = {"rows": len(df), "policies": counts, "all_scenarios": len(all_ids),
             "architecture_feasible_scenarios": len(feasible_ids),
             "architecture_feasible_ids": sorted(feasible_ids), "bootstrap_resamples": N_BOOT,
             "random_seed": 20260911, "temperature_violations_total": int(df.violation_steps.sum())}
    with (OUT / "unified_final_test_audit_v02.json").open("w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print(result.to_string(index=False))
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print("UNIFIED_FINAL_TEST_ANALYSIS_V02_COMPLETE")


if __name__ == "__main__":
    main()
