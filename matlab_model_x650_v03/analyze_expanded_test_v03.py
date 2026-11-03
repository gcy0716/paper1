from pathlib import Path
import csv
import math
import sys
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
tag = sys.argv[1] if len(sys.argv) > 1 else "finetuned"
SOURCE = RESULTS / f"sac_{tag}_test_comparison_v03.csv"
rows = list(csv.DictReader(SOURCE.open(encoding="utf-8-sig")))

metrics = [
    "peak_temperature_C", "total_reward", "fan_energy_Wh", "hydrogen_g",
    "unmet_energy_kJ", "final_SOC", "violation_steps",
    "safety_override_steps", "action_total_variation",
    "mission_energy_adequacy", "equivalent_fully_served_time_min",
    "unmet_equivalent_time_min", "served_energy_Wh_per_kg",
]
by_policy: dict[str, list[dict[str, str]]] = defaultdict(list)
for row in rows:
    by_policy[row["policy"]].append(row)

assert len(rows) == 40, f"expected 4 policies x 10 scenarios, got {len(rows)}"
assert all(len(v) == 10 for v in by_policy.values())

summary_rows = []
for policy, policy_rows in sorted(by_policy.items()):
    for metric in metrics:
        x = np.array([float(r[metric]) for r in policy_rows])
        std = float(x.std(ddof=1))
        ci = 1.96 * std / math.sqrt(len(x))
        summary_rows.append({
            "policy": policy, "metric": metric, "n": len(x),
            "mean": float(x.mean()), "std": std,
            "ci95_low": float(x.mean() - ci), "ci95_high": float(x.mean() + ci),
        })

with (RESULTS / f"sac_{tag}_expanded_statistics_v03.csv").open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
    writer.writeheader(); writer.writerows(summary_rows)

indexed = {(r["policy"], int(float(r["seed"]))): r for r in rows}
sac_name = next(p for p in by_policy if p.startswith("sac_"))
paired_rows = []
for baseline in ("mpc", "pi"):
    for metric in metrics:
        delta = np.array([
            float(indexed[(sac_name, seed)][metric]) - float(indexed[(baseline, seed)][metric])
            for seed in sorted(int(float(r["seed"])) for r in by_policy[sac_name])
        ])
        std = float(delta.std(ddof=1)); ci = 1.96 * std / math.sqrt(len(delta))
        paired_rows.append({
            "comparison": f"{sac_name}_minus_{baseline}", "metric": metric,
            "n": len(delta), "mean_delta": float(delta.mean()), "std_delta": std,
            "ci95_low": float(delta.mean() - ci), "ci95_high": float(delta.mean() + ci),
        })

with (RESULTS / f"sac_{tag}_paired_differences_v03.csv").open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=paired_rows[0].keys())
    writer.writeheader(); writer.writerows(paired_rows)

assert all(float(r["violation_steps"]) == 0 for r in rows)
assert all(float(r["safety_override_steps"]) == 0 for r in rows)

def paired(comparison: str, metric: str):
    return next(r for r in paired_rows if r["comparison"] == comparison and r["metric"] == metric)

report = RESULTS / f"sac_{tag}_expanded_audit_v03.txt"
key = f"{sac_name}_minus_mpc"
report.write_text(
    "EXPANDED_TEST_PASS\n"
    "matched_scenarios=10\n"
    "policies=4\n"
    "constraint_violations=0\n"
    f"sac_minus_mpc_peak_temperature_C={paired(key, 'peak_temperature_C')['mean_delta']:.6f}\n"
    f"sac_minus_mpc_fan_energy_Wh={paired(key, 'fan_energy_Wh')['mean_delta']:.6f}\n"
    f"sac_minus_mpc_total_reward={paired(key, 'total_reward')['mean_delta']:.6f}\n"
    f"sac_minus_mpc_mission_energy_adequacy={paired(key, 'mission_energy_adequacy')['mean_delta']:.6f}\n"
    f"sac_minus_mpc_equivalent_fully_served_time_min={paired(key, 'equivalent_fully_served_time_min')['mean_delta']:.6f}\n",
    encoding="utf-8",
)
print(report.read_text(encoding="utf-8"))
