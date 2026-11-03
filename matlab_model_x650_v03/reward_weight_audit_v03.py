from pathlib import Path
import csv
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
rows = list(csv.DictReader((RESULTS / "sac_finetuned_test_comparison_v03.csv").open(encoding="utf-8-sig")))

weight_sets = [
    ("temperature_focused", 1.0, 0.08, 0.05),
    ("current_candidate", 1.0, 0.12, 0.08),
    ("energy_focused", 1.0, 0.20, 0.08),
    ("smoothness_focused", 1.0, 0.12, 0.15),
]

audit = []
for name, wt, wf, ws in weight_sets:
    values = defaultdict(list)
    for r in rows:
        score = -(
            wt * float(r["temperature_cost_sum"])
            + wf * float(r["fan_cost_sum"])
            + ws * float(r["smooth_cost_sum"])
            + 100.0 * float(r["violation_steps"])
            + 0.002 * float(r["unmet_cost_sum"])
        )
        values[r["policy"]].append(score)
    ranking = sorted(((float(np.mean(v)), p) for p, v in values.items()), reverse=True)
    for rank, (mean_score, policy) in enumerate(ranking, 1):
        audit.append({"weight_set": name, "policy": policy, "rank": rank,
                      "mean_recomputed_reward": mean_score})

with (RESULTS / "reward_weight_posthoc_audit_v03.csv").open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=audit[0].keys())
    writer.writeheader(); writer.writerows(audit)

winner = {name: next(r["policy"] for r in audit if r["weight_set"] == name and r["rank"] == 1)
          for name, *_ in weight_sets}
report = RESULTS / "reward_weight_posthoc_audit_v03.txt"
report.write_text(
    "POSTHOC_REWARD_WEIGHT_AUDIT_PASS\n"
    + "\n".join(f"{k}_winner={v}" for k, v in winner.items())
    + "\nwarning=this_is_posthoc_scoring_not_retraining_sensitivity\n",
    encoding="utf-8",
)
print(report.read_text(encoding="utf-8"))
