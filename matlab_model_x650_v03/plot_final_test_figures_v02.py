from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "predictive_safety_v02"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(exist_ok=True)

episodes = pd.read_csv(RESULTS / "unified_final_test_episodes_v02.csv")
boot = pd.read_csv(RESULTS / "unified_final_test_bootstrap_v02.csv")

# The feasibility definition was fixed in the statistical audit: both safe-PI
# and MPC must have unmet energy below 0.1 kJ in the same scenario.
pivot_unmet = episodes.pivot_table(
    index="scenario_seed", columns="policy", values="unmet_energy_kJ", aggfunc="first"
)
feasible_seeds = pivot_unmet.index[
    (pivot_unmet["safe_pi"] < 0.1) & (pivot_unmet["mpc"] < 0.1)
]
feasible = episodes[episodes["scenario_seed"].isin(feasible_seeds)].copy()

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 160,
        "savefig.dpi": 600,
    }
)


def save(fig, stem):
    fig.savefig(FIGURES / f"{stem}.png", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


# Figure 1: paired P90-versus-legacy effects on architecture-feasible scenarios.
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
colors = {11: "#2C7FB8", 22: "#41AB5D", 33: "#D95F0E"}
for ax, metric, ylabel in [
    (axes[0], "peak_temperature_C", "Peak temperature (°C)"),
    (axes[1], "fan_energy_Wh", "Fan energy (Wh)"),
]:
    for seed in (11, 22, 33):
        left = feasible[feasible.policy == f"sac_seed{seed}_legacy"].set_index("scenario_seed")[metric]
        right = feasible[feasible.policy == f"sac_seed{seed}_p90"].set_index("scenario_seed")[metric]
        common = left.index.intersection(right.index)
        jitter = (seed - 22) / 90
        for sid in common:
            ax.plot([0 + jitter, 1 + jitter], [left[sid], right[sid]], color=colors[seed], alpha=0.18, lw=0.7)
        ax.scatter(np.full(len(common), 0 + jitter), left.loc[common], s=9, color=colors[seed], alpha=0.5)
        ax.scatter(np.full(len(common), 1 + jitter), right.loc[common], s=9, color=colors[seed], alpha=0.5)
        ax.plot([0 + jitter, 1 + jitter], [left.mean(), right.mean()], color=colors[seed], lw=2.2,
                marker="o", ms=4, label=f"SAC seed {seed}")
    ax.set_xticks([0, 1], ["Legacy", "P90"])
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.2)
axes[0].axhline(65, color="#B2182B", ls="--", lw=1.1, label="Hard limit")
axes[0].set_title("(a) Thermal effect")
axes[1].set_title("(b) Auxiliary-energy cost")
handles, labels_legend = axes[0].get_legend_handles_labels()
fig.legend(handles, labels_legend, frameon=False, fontsize=7, ncol=4,
           loc="upper center", bbox_to_anchor=(0.5, 0.94))
fig.suptitle("Paired effect of the predictive safety layer (n = 15 scenarios)", y=1.04, fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.88))
save(fig, "Fig_predictive_safety_paired_v02")


# Figure 2: controller Pareto map on architecture-feasible scenarios.
order = ["mpc", "safe_pi", "sac_seed11_p90", "sac_seed22_p90", "sac_seed33_p90"]
labels = {"mpc": "MPC", "safe_pi": "Safe PI", "sac_seed11_p90": "SAC-11 + P90",
          "sac_seed22_p90": "SAC-22 + P90", "sac_seed33_p90": "SAC-33 + P90"}
palette = {"mpc": "#2166AC", "safe_pi": "#878787", "sac_seed11_p90": "#67A9CF",
           "sac_seed22_p90": "#FDAE61", "sac_seed33_p90": "#D73027"}
fig, ax = plt.subplots(figsize=(5.2, 3.8))
for policy in order:
    group = feasible[feasible.policy == policy]
    x = group["fan_energy_Wh"].mean()
    y = group["peak_temperature_C"].mean()
    xerr = group["fan_energy_Wh"].std(ddof=1) / np.sqrt(len(group))
    yerr = group["peak_temperature_C"].std(ddof=1) / np.sqrt(len(group))
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", ms=6, capsize=2.5,
                color=palette[policy], label=labels[policy])
    ax.annotate(labels[policy], (x, y), xytext=(5, 4), textcoords="offset points", fontsize=7)
ax.axhline(65, color="#B2182B", ls="--", lw=1.1)
ax.set_xlabel("Mean fan energy (Wh)")
ax.set_ylabel("Mean peak temperature (°C)")
ax.set_title("Temperature–auxiliary-energy trade-off\n(architecture-feasible final-test scenarios)")
ax.grid(alpha=0.2)
fig.tight_layout()
save(fig, "Fig_controller_pareto_v02")


# Machine-readable values used by the plots.
rows = []
for policy in order:
    group = feasible[feasible.policy == policy]
    rows.append(
        {
            "policy": policy,
            "n_scenarios": len(group),
            "mean_peak_temperature_C": group.peak_temperature_C.mean(),
            "sem_peak_temperature_C": group.peak_temperature_C.sem(),
            "mean_fan_energy_Wh": group.fan_energy_Wh.mean(),
            "sem_fan_energy_Wh": group.fan_energy_Wh.sem(),
            "total_violation_steps": int(group.violation_steps.sum()),
        }
    )
pd.DataFrame(rows).to_csv(RESULTS / "final_test_figure_values_v02.csv", index=False)
print(f"Saved figures and source values to {FIGURES}")
