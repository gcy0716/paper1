from pathlib import Path
import csv
import math
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
TRAINING_SEEDS = (11, 22, 33)
TEST_SCENARIOS = 10
MISSION_MIN = 20.0
TAKEOFF_MASS_KG = 5.2
BATTERY_EUSABLE_WH = 22.2 * 5
SOC_WINDOW = 0.80 - 0.25
BATTERY_DISCHARGE_EFFICIENCY = 0.95
DCDC_EFFICIENCY = 0.94


def read_rows(seed: int):
    path = OUT / f"sac_seed{seed}_test_comparison_v03.csv"
    return list(csv.DictReader(path.open(encoding="utf-8-sig")))


files = {seed: read_rows(seed) for seed in TRAINING_SEEDS}
baseline = [r for r in files[11] if r["policy"] in ("fixed", "pi", "mpc")]
sac = []
for training_seed, rows in files.items():
    for row in rows:
        if row["policy"].startswith("sac_"):
            row = dict(row)
            row["training_seed"] = str(training_seed)
            row["policy_group"] = "sac"
            sac.append(row)

groups = {
    "fixed": [r for r in baseline if r["policy"] == "fixed"],
    "pi": [r for r in baseline if r["policy"] == "pi"],
    "mpc": [r for r in baseline if r["policy"] == "mpc"],
    "sac": sac,
}
metrics = (
    "peak_temperature_C", "fan_energy_Wh", "unmet_energy_kJ",
    "mission_energy_adequacy", "equivalent_fully_served_time_min",
    "demand_energy_Wh", "total_load_energy_Wh", "served_load_energy_Wh",
    "fc_dc_energy_Wh", "battery_discharge_energy_Wh", "served_energy_Wh_per_kg",
)

summary = []
for policy, rows in groups.items():
    record = {"policy": policy, "n": len(rows)}
    for metric in metrics:
        values = np.array([float(r[metric]) for r in rows])
        record[f"mean_{metric}"] = float(values.mean())
        record[f"std_{metric}"] = float(values.std(ddof=1))
    summary.append(record)

summary_path = OUT / "system_feasibility_summary_v03.csv"
with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
    writer.writeheader()
    writer.writerows(summary)

mpc_by_scenario = {int(float(r["seed"])): r for r in groups["mpc"]}
paired = []
for training_seed in TRAINING_SEEDS:
    rows = [r for r in sac if int(r["training_seed"]) == training_seed]
    for metric in ("peak_temperature_C", "fan_energy_Wh", "unmet_energy_kJ",
                   "mission_energy_adequacy", "equivalent_fully_served_time_min"):
        delta = np.array([
            float(r[metric]) - float(mpc_by_scenario[int(float(r["seed"]))][metric])
            for r in rows
        ])
        ci = 1.96 * delta.std(ddof=1) / math.sqrt(len(delta))
        paired.append({
            "training_seed": training_seed,
            "comparison": "sac_minus_mpc",
            "metric": metric,
            "n": len(delta),
            "mean_delta": float(delta.mean()),
            "ci95_low": float(delta.mean() - ci),
            "ci95_high": float(delta.mean() + ci),
        })

paired_path = OUT / "system_feasibility_paired_sac_mpc_v03.csv"
with paired_path.open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=paired[0].keys())
    writer.writeheader()
    writer.writerows(paired)

mpc = next(r for r in summary if r["policy"] == "mpc")
sac_mean = next(r for r in summary if r["policy"] == "sac")
deficit_wh = mpc["mean_unmet_energy_kJ"] / 3.6
fan_fraction = mpc["mean_fan_energy_Wh"] / mpc["mean_total_load_energy_Wh"]
extra_fc_dc_w = deficit_wh / (MISSION_MIN / 60)
extra_fc_stack_w = extra_fc_dc_w / DCDC_EFFICIENCY
extra_battery_parameter_wh = deficit_wh / (SOC_WINDOW * BATTERY_DISCHARGE_EFFICIENCY)
required_battery_parameter_wh = BATTERY_EUSABLE_WH + extra_battery_parameter_wh
required_battery_ah_6s = required_battery_parameter_wh / 22.2
sac_peak_delta = sac_mean["mean_peak_temperature_C"] - mpc["mean_peak_temperature_C"]
sac_fan_delta = sac_mean["mean_fan_energy_Wh"] - mpc["mean_fan_energy_Wh"]
sac_adequacy_delta = sac_mean["mean_mission_energy_adequacy"] - mpc["mean_mission_energy_adequacy"]

diagnosis = f"""# v03 system-level feasibility diagnosis

## Scope

This report uses three independently trained SAC policies and ten matched high-altitude test scenarios per policy. Fixed, PI and MPC baselines use the same scenario seeds. Results remain simulation-only and depend on the current assumed mission, mass and component parameters.

## Main finding

The dominant limitation is system energy capacity rather than fan-control parasitic power.

- Mean 20 min total load energy under MPC: {mpc['mean_total_load_energy_Wh']:.2f} Wh.
- Mean FC energy delivered through the DC/DC: {mpc['mean_fc_dc_energy_Wh']:.2f} Wh.
- Mean battery discharge delivered to the bus: {mpc['mean_battery_discharge_energy_Wh']:.2f} Wh.
- Mean unmet energy: {deficit_wh:.2f} Wh ({mpc['mean_unmet_energy_kJ']:.2f} kJ).
- Mean mission-energy adequacy: {100*mpc['mean_mission_energy_adequacy']:.2f}%.
- Fan energy is only {mpc['mean_fan_energy_Wh']:.2f} Wh, or {100*fan_fraction:.2f}% of total load energy.

Even eliminating fan power entirely would recover less than {100*mpc['mean_fan_energy_Wh']/deficit_wh:.1f}% of the present energy deficit. Thermal-control optimization therefore cannot by itself restore mission feasibility.

## SAC versus MPC

Across all three training seeds, the SAC mean relative to MPC is:

- Peak stack temperature: {sac_peak_delta:+.2f} degC.
- Fan energy: {sac_fan_delta:+.3f} Wh.
- Mission-energy adequacy: {100*sac_adequacy_delta:+.3f} percentage points.

This supports a trade-off interpretation, not a claim that SAC dominates MPC. SAC policies occupy nearby temperature-energy operating points, while architecture-level sizing controls the much larger mission-energy deficit.

## Preliminary sizing bounds

These are first-order energy-balance bounds, not dynamic redesign results.

- If the deficit is covered only by continuous FC power, the bus needs about {extra_fc_dc_w:.0f} W additional average power, corresponding to about {extra_fc_stack_w:.0f} W before the current DC/DC efficiency. This suggests testing at least 1.3, 1.5 and 1.6 kW FC ratings.
- If the deficit is covered only by the buffer battery while retaining the current SOC window and discharge efficiency, the battery energy parameter rises from {BATTERY_EUSABLE_WH:.1f} to about {required_battery_parameter_wh:.1f} Wh, equivalent to about {required_battery_ah_6s:.1f} Ah at 22.2 V. This does not include the associated mass increase.
- A mixed FC-battery redesign is preferable because it can address both sustained high-altitude demand and transient power while preserving mass margin.

## Required next experiment

Run a common architecture-control matrix over FC rating, battery energy, takeoff mass and controller. Each case must report temperature safety, energy adequacy, final SOC, unmet energy, hydrogen use, fan energy and power-to-weight ratio. Architectures with any unmet energy or SOC-floor contact should be classified as mission-infeasible before controller ranking.
"""
(OUT / "SYSTEM_FEASIBILITY_DIAGNOSIS_v03.md").write_text(diagnosis, encoding="utf-8")

try:
    import matplotlib.pyplot as plt
    colors = {"fixed": "#7f8c8d", "pi": "#d95f02", "mpc": "#1b9e77", "sac": "#7570b3"}
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    for row in summary:
        ax.scatter(row["mean_fan_energy_Wh"], row["mean_peak_temperature_C"],
                   s=90, color=colors[row["policy"]], label=row["policy"].upper())
        ax.annotate(f"{row['policy'].upper()}\nadequacy={100*row['mean_mission_energy_adequacy']:.1f}%",
                    (row["mean_fan_energy_Wh"], row["mean_peak_temperature_C"]),
                    xytext=(6, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Mean fan energy (Wh, 20 min mission)")
    ax.set_ylabel("Mean peak stack temperature (degC)")
    ax.set_title("Thermal-control trade-off under matched high-altitude scenarios")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(OUT / "thermal_energy_pareto_v03.png", dpi=220)
    plt.close(fig)
except ImportError:
    pass

print(diagnosis)
