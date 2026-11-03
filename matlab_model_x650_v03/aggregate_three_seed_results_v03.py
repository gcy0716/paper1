from pathlib import Path
import csv
import math
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
seeds = (11, 22, 33)
all_sac = []
baseline = None
for training_seed in seeds:
    path = OUT / f"sac_seed{training_seed}_test_comparison_v03.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    if baseline is None:
        baseline = [r for r in rows if r["policy"] in ("fixed", "pi", "mpc")]
    for r in rows:
        if r["policy"].startswith("sac_"):
            r["training_seed"] = str(training_seed)
            all_sac.append(r)

assert len(all_sac) == 30
metrics = ["peak_temperature_C", "total_reward", "fan_energy_Wh", "hydrogen_g",
           "unmet_energy_kJ", "final_SOC", "violation_steps", "safety_override_steps",
           "action_total_variation", "mission_energy_adequacy",
           "equivalent_fully_served_time_min", "unmet_equivalent_time_min",
           "served_energy_Wh_per_kg"]

with (OUT / "sac_three_seed_all_test_episodes_v03.csv").open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=all_sac[0].keys())
    writer.writeheader(); writer.writerows(all_sac)

summary = []
for metric in metrics:
    x = np.array([float(r[metric]) for r in all_sac])
    std = float(x.std(ddof=1)); ci = 1.96 * std / math.sqrt(len(x))
    summary.append({"metric": metric, "n": len(x), "mean": float(x.mean()), "std": std,
                    "ci95_low": float(x.mean()-ci), "ci95_high": float(x.mean()+ci)})
with (OUT / "sac_three_seed_overall_statistics_v03.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w=csv.DictWriter(f,fieldnames=summary[0].keys()); w.writeheader(); w.writerows(summary)

# Seed-level means reveal training reproducibility, without treating 30 episodes
# as 30 independent training replications.
seed_summary=[]
for seed in seeds:
    rr=[r for r in all_sac if int(r["training_seed"])==seed]
    row={"training_seed":seed}
    for metric in metrics: row[f"mean_{metric}"]=float(np.mean([float(r[metric]) for r in rr]))
    seed_summary.append(row)
with (OUT / "sac_three_seed_seedlevel_means_v03.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w=csv.DictWriter(f,fieldnames=seed_summary[0].keys()); w.writeheader(); w.writerows(seed_summary)

# 95% t interval across the three independent training seeds (df=2).
tcrit_df2 = 4.3026527299
seed_inference=[]
for metric in metrics:
    x=np.array([r[f"mean_{metric}"] for r in seed_summary])
    std=float(x.std(ddof=1)); ci=tcrit_df2*std/math.sqrt(len(x))
    seed_inference.append({"metric":metric,"training_seeds":3,"mean":float(x.mean()),
                           "between_seed_std":std,"ci95_low":float(x.mean()-ci),
                           "ci95_high":float(x.mean()+ci)})
with (OUT / "sac_three_seed_training_seed_inference_v03.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w=csv.DictWriter(f,fieldnames=seed_inference[0].keys()); w.writeheader(); w.writerows(seed_inference)

# Paired comparisons retain scenario identity.  Baseline episodes are shared
# across training seeds; each independently trained SAC is compared against
# the same baseline response for the same test seed.
baseline_index={(r["policy"],int(float(r["seed"]))):r for r in baseline}
paired=[]
for training_seed in seeds:
    sac_rows=[r for r in all_sac if int(r["training_seed"])==training_seed]
    for base in ("fixed","pi","mpc"):
        for metric in metrics:
            delta=np.array([float(r[metric])-float(baseline_index[(base,int(float(r["seed"])) )][metric]) for r in sac_rows])
            std=float(delta.std(ddof=1)); ci=1.96*std/math.sqrt(len(delta))
            paired.append({"training_seed":training_seed,"comparison":f"sac_minus_{base}",
                           "metric":metric,"n":len(delta),"mean_delta":float(delta.mean()),
                           "std_delta":std,"ci95_low":float(delta.mean()-ci),
                           "ci95_high":float(delta.mean()+ci)})
with (OUT / "sac_three_seed_paired_baseline_differences_v03.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w=csv.DictWriter(f,fieldnames=paired[0].keys()); w.writeheader(); w.writerows(paired)

assert all(float(r["violation_steps"]) == 0 for r in all_sac)
assert all(float(r["safety_override_steps"]) == 0 for r in all_sac)
report=OUT/"sac_three_seed_audit_v03.txt"
def mean(metric): return next(r["mean"] for r in summary if r["metric"]==metric)
report.write_text(
    "THREE_SEED_AUDIT_PASS\ntraining_seeds=3\ntest_scenarios_per_seed=10\n"
    "total_sac_test_episodes=30\nconstraint_violations=0\n"
    f"mean_peak_temperature_C={mean('peak_temperature_C'):.6f}\n"
    f"mean_fan_energy_Wh={mean('fan_energy_Wh'):.6f}\n"
    f"mean_action_total_variation={mean('action_total_variation'):.6f}\n",
    encoding="utf-8")
print(report.read_text(encoding="utf-8"))
