from pathlib import Path
import csv
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
SOURCE = OUT / "architecture_screen_v03.csv"
rows = list(csv.DictReader(SOURCE.open(encoding="utf-8-sig")))


def number(row, key):
    try:
        return float(row[key])
    except ValueError:
        return float("nan")


def truth(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


by_arch = defaultdict(list)
for row in rows:
    key = (int(number(row, "fc_rating_W")), number(row, "battery_capacity_Ah"))
    by_arch[key].append(row)

architecture_rows = []
for (rating, capacity), group in sorted(by_arch.items()):
    mass_ok = all(truth(r["mass_feasible"]) for r in group)
    valid = [r for r in group if np.isfinite(number(r, "final_SOC"))]
    all_controllers_ok = bool(valid) and all(truth(r["mission_feasible"]) for r in group)
    if valid:
        worst_final_soc = min(number(r, "final_SOC") for r in valid)
        worst_peak_temp = max(number(r, "peak_stack_temperature_C") for r in valid)
        worst_unmet = max(number(r, "unmet_energy_kJ") for r in valid)
        mean_h2 = float(np.mean([number(r, "hydrogen_used_g") for r in valid]))
        takeoff_mass = number(valid[0], "takeoff_mass_kg")
        power_to_weight = number(valid[0], "fc_power_to_weight_kW_per_kg")
    else:
        worst_final_soc = worst_peak_temp = worst_unmet = mean_h2 = float("nan")
        takeoff_mass = number(group[0], "takeoff_mass_kg")
        power_to_weight = number(group[0], "fc_power_to_weight_kW_per_kg")
    architecture_rows.append({
        "fc_rating_W": rating,
        "battery_capacity_Ah": capacity,
        "takeoff_mass_kg": takeoff_mass,
        "fc_power_to_weight_kW_per_kg": power_to_weight,
        "mass_feasible": mass_ok,
        "all_controllers_mission_feasible": all_controllers_ok,
        "worst_controller_final_SOC": worst_final_soc,
        "worst_controller_peak_temperature_C": worst_peak_temp,
        "worst_controller_unmet_energy_kJ": worst_unmet,
        "mean_hydrogen_used_g": mean_h2,
    })

feasible = [r for r in architecture_rows if r["all_controllers_mission_feasible"]]
feasible.sort(key=lambda r: (r["takeoff_mass_kg"], -r["worst_controller_final_SOC"]))
for rank, row in enumerate(feasible, 1):
    row["minimum_mass_rank"] = rank
for row in architecture_rows:
    row.setdefault("minimum_mass_rank", "")

out_csv = OUT / "architecture_level_screen_v03.csv"
with out_csv.open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=architecture_rows[0].keys())
    writer.writeheader()
    writer.writerows(architecture_rows)

minimum = feasible[0]
balanced = next(r for r in feasible if r["fc_rating_W"] == 1500 and r["battery_capacity_Ah"] == 5)
report = f"""# 架构筛选结论（v03）

## 筛选范围与约束

- PEMFC额定功率：1.1、1.3、1.5、1.6 kW；
- 6S缓冲电池：5、7.5、10 Ah；
- 控制器：固定风机、安全PI、MPC；
- 名义筛选高度：6000 m；
- 通过条件：起飞质量不超过6.3 kg、未满足能量小于0.1 kJ、末端SOC高于0.26、峰值电堆温度不超过65 degC；
- 电堆比功率、锂电池比能量及模块化热参数缩放均为概念设计假设，不是实测值。

## 架构级结果

共有{len(feasible)}套架构能使三种控制器全部通过当前名义任务约束。

最低质量可行架构为：

- PEMFC：{minimum['fc_rating_W']/1000:.1f} kW；
- 缓冲电池：{minimum['battery_capacity_Ah']:.1f} Ah；
- 起飞质量：{minimum['takeoff_mass_kg']:.3f} kg；
- 最差控制器末端SOC：{minimum['worst_controller_final_SOC']:.3f}；
- 最差峰值温度：{minimum['worst_controller_peak_temperature_C']:.2f} degC；
- 最大未满足能量：{minimum['worst_controller_unmet_energy_kJ']:.3f} kJ。

推荐作为下一轮鲁棒性测试中心点的折中架构为：

- PEMFC：1.5 kW；
- 缓冲电池：5 Ah；
- 起飞质量：{balanced['takeoff_mass_kg']:.3f} kg；
- 最差控制器末端SOC：{balanced['worst_controller_final_SOC']:.3f}；
- PEMFC额定功率/起飞质量：{balanced['fc_power_to_weight_kW_per_kg']:.3f} kW/kg。

1.3 kW + 5 Ah适合定义为“最低质量名义可行解”，但其SOC和功率裕度不宜直接视为鲁棒。1.5 kW + 5 Ah质量仅增加约{balanced['takeoff_mass_kg']-minimum['takeoff_mass_kg']:.3f} kg，却显著提高SOC余量，因此更适合作为后续高空扰动、参数不确定性和SAC重新训练的中心架构。

## 电池增容反效果

在当前比能量和旋翼功率质量指数假设下，增加电池容量并不保证提高任务裕度。电池增重会提高悬停与爬升功率，部分组合中新增需求超过新增可用电池能量。例如1.3 kW架构从5 Ah增加到7.5 Ah后，名义末端SOC反而逼近下限。

这不是普适结论，而是一个必须进行比能量敏感性分析的模型发现。正式论文应报告不同电池比能量、质量指数和任务剖面下该结论是否保持。

## 下一步质量门槛

当前筛选仅代表6000 m名义场景。推荐架构在进入论文结论前还必须通过：

1. 5000--6000 m随机海拔和环境温度；
2. 负载、换热系数和热容扰动；
3. PEMFC降额和DC/DC效率扰动；
4. 多随机种子SAC重新训练；
5. 所有控制器使用完全相同的未见场景进行配对统计。
"""
(OUT / "架构筛选结论_v03.md").write_text(report, encoding="utf-8")

try:
    import matplotlib.pyplot as plt
    mpc = [r for r in rows if r["controller"] == "mpc" and np.isfinite(number(r, "final_SOC"))]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for rating in sorted({int(number(r, "fc_rating_W")) for r in mpc}):
        subset = sorted([r for r in mpc if int(number(r, "fc_rating_W")) == rating],
                        key=lambda r: number(r, "battery_capacity_Ah"))
        axes[0].plot([number(r, "takeoff_mass_kg") for r in subset],
                     [number(r, "final_SOC") for r in subset], marker="o", label=f"{rating/1000:.1f} kW")
        for r in subset:
            axes[0].annotate(f"{number(r, 'battery_capacity_Ah'):g} Ah",
                             (number(r, "takeoff_mass_kg"), number(r, "final_SOC")),
                             xytext=(4, 4), textcoords="offset points", fontsize=8)
    axes[0].axhline(0.26, color="crimson", linestyle="--", label="SOC feasibility margin")
    axes[0].axvline(6.3, color="black", linestyle=":", label="mass limit")
    axes[0].set(xlabel="Takeoff mass (kg)", ylabel="Final SOC",
                title="Mass-capacity coupling at 6000 m (MPC)")
    axes[0].grid(alpha=.25); axes[0].legend(frameon=False, fontsize=8)

    valid_arch = [r for r in architecture_rows if r["mass_feasible"]]
    colors = ["#1b9e77" if r["all_controllers_mission_feasible"] else "#d95f02" for r in valid_arch]
    axes[1].scatter([r["takeoff_mass_kg"] for r in valid_arch],
                    [r["fc_power_to_weight_kW_per_kg"] for r in valid_arch],
                    c=colors, s=70)
    for r in valid_arch:
        axes[1].annotate(f"{r['fc_rating_W']/1000:.1f}kW/{r['battery_capacity_Ah']:g}Ah",
                         (r["takeoff_mass_kg"], r["fc_power_to_weight_kW_per_kg"]),
                         xytext=(4, 3), textcoords="offset points", fontsize=7)
    axes[1].set(xlabel="Takeoff mass (kg)", ylabel="FC rated power / takeoff mass (kW/kg)",
                title="Architecture feasibility (green = all controllers pass)")
    axes[1].grid(alpha=.25)
    fig.savefig(OUT / "architecture_screen_v03.png", dpi=220)
    plt.close(fig)
except ImportError:
    pass

print(report)
