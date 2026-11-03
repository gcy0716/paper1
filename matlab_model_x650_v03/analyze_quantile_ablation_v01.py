from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "quantile_ablation_v01"
df = pd.read_csv(OUT / "quantile_ablation_episodes_v01.csv")
rng = np.random.default_rng(20260911)
B = 10_000

# Architecture-feasible means no variant of any frozen SAC seed has material
# unmet energy in that scenario. This prevents auxiliary-load competition from
# being misread as a thermal-controller effect.
max_unmet = df.groupby("scenario_seed").unmet_energy_kJ.max()
feasible_seeds = max_unmet[max_unmet < 0.1].index

comparisons = [("legacy", "p50"), ("legacy", "p90"), ("p50", "p90")]
metrics = ["peak_temperature_C", "fan_energy_Wh", "unmet_energy_kJ"]
rows = []
for scope, data in [("all", df), ("architecture_feasible", df[df.scenario_seed.isin(feasible_seeds)])]:
    for seed in (11, 22, 33):
        sd = data[data.agent_seed == seed]
        for left, right in comparisons:
            for metric in metrics:
                a = sd[sd.variant == left].set_index("scenario_seed")[metric]
                b = sd[sd.variant == right].set_index("scenario_seed")[metric]
                common = a.index.intersection(b.index)
                delta = (b.loc[common] - a.loc[common]).to_numpy()
                boot = delta[rng.integers(0, len(delta), size=(B, len(delta)))].mean(axis=1)
                rows.append({
                    "scope": scope, "agent_seed": seed, "left": left, "right": right,
                    "metric": metric, "n_scenarios": len(delta),
                    "mean_delta_right_minus_left": delta.mean(),
                    "bootstrap_ci95_low": np.quantile(boot, .025),
                    "bootstrap_ci95_high": np.quantile(boot, .975),
                })

stats = pd.DataFrame(rows)
stats.to_csv(OUT / "quantile_ablation_bootstrap_v01.csv", index=False)
audit = {
    "rows": int(len(df)), "expected_rows": 180,
    "scenario_count": int(df.scenario_seed.nunique()),
    "policies_per_scenario_min": int(df.groupby("scenario_seed").size().min()),
    "policies_per_scenario_max": int(df.groupby("scenario_seed").size().max()),
    "architecture_feasible_scenarios": int(len(feasible_seeds)),
    "architecture_feasible_seeds": [int(x) for x in feasible_seeds],
    "total_temperature_violation_steps": int(df.violation_steps.sum()),
    "bootstrap_resamples": B,
    "final_test_seeds_present": bool(df.scenario_seed.between(3101, 3120).any()),
}
(OUT / "quantile_ablation_audit_v01.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

# Forest plot of thermal and auxiliary-energy effects in feasible scenarios.
plot_data = stats[(stats.scope == "architecture_feasible") &
                  (stats.left == "legacy") & (stats.right.isin(["p50", "p90"]))]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
colors = {"p50": "#0077BB", "p90": "#EE7733"}
for ax, metric, xlabel in [
    (axes[0], "peak_temperature_C", "Change in peak temperature (°C)"),
    (axes[1], "fan_energy_Wh", "Change in fan energy (Wh)"),
]:
    d = plot_data[plot_data.metric == metric]
    ypos = []
    ylabels = []
    y = 0
    for seed in (11, 22, 33):
        for variant in ("p50", "p90"):
            row = d[(d.agent_seed == seed) & (d.right == variant)].iloc[0]
            ax.errorbar(row.mean_delta_right_minus_left, y,
                        xerr=[[row.mean_delta_right_minus_left-row.bootstrap_ci95_low],
                              [row.bootstrap_ci95_high-row.mean_delta_right_minus_left]],
                        fmt="o" if variant == "p50" else "s", color=colors[variant], capsize=2.5)
            ypos.append(y); ylabels.append(f"Seed {seed}, {variant.upper()}"); y += 1
        y += .35
    ax.axvline(0, color="#555555", ls="--", lw=1)
    ax.set_yticks(ypos, ylabels if ax is axes[0] else [])
    ax.set_xlabel(xlabel); ax.grid(axis="x", alpha=.2); ax.invert_yaxis()
axes[0].set_title("(a) Thermal benefit")
axes[1].set_title("(b) Auxiliary-energy cost")
fig.suptitle("Risk-quantile ablation versus reactive protection", fontsize=10)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"Fig_quantile_ablation_v01.{ext}", dpi=600, bbox_inches="tight")
plt.close(fig)
print(json.dumps(audit, ensure_ascii=False))
