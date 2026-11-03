from pathlib import Path
import csv, math
import numpy as np

ROOT=Path(__file__).resolve().parent; OUT=ROOT/"results"; seeds=(11,22,33)
all_sac=[]; baseline=None
for seed in seeds:
    rows=list(csv.DictReader((OUT/f"sac_v05_pilot_seed{seed}_test_comparison_v03.csv").open(encoding="utf-8-sig")))
    if baseline is None: baseline=[r for r in rows if r["policy"] in ("fixed","pi","mpc")]
    for row in rows:
        if row["policy"].startswith("sac_"):
            row=dict(row); row["training_seed"]=str(seed); all_sac.append(row)

metrics=("peak_temperature_C","total_reward","fan_energy_Wh","hydrogen_g","unmet_energy_kJ",
         "final_SOC","violation_steps","safety_override_steps","action_total_variation","mission_energy_adequacy")
base_index={(r["policy"],int(float(r["seed"]))):r for r in baseline}
seed_means=[]; paired=[]
for seed in seeds:
    rr=[r for r in all_sac if int(r["training_seed"])==seed]; rec={"training_seed":seed}
    for metric in metrics:
        rec[f"mean_{metric}"]=float(np.mean([float(r[metric]) for r in rr]))
        d=np.array([float(r[metric])-float(base_index[("mpc",int(float(r["seed"])))][metric]) for r in rr])
        ci=1.96*d.std(ddof=1)/math.sqrt(len(d))
        paired.append({"training_seed":seed,"comparison":"sac_minus_mpc","metric":metric,"n":len(d),
            "mean_delta":float(d.mean()),"ci95_low":float(d.mean()-ci),"ci95_high":float(d.mean()+ci)})
    rec["strict_feasible_scenarios"]=sum(float(r["unmet_energy_kJ"])<.1 and float(r["final_SOC"])>.26 and float(r["violation_steps"])==0 for r in rr)
    seed_means.append(rec)

for name,data in (("v05_pilot_all_sac_episodes.csv",all_sac),("v05_pilot_seed_means.csv",seed_means),
                  ("v05_pilot_paired_differences.csv",paired)):
    with (OUT/name).open("w",newline="",encoding="utf-8-sig") as h:
        w=csv.DictWriter(h,fieldnames=data[0].keys()); w.writeheader(); w.writerows(data)

old=list(csv.DictReader((OUT/"v05_transfer_seed_means.csv").open(encoding="utf-8-sig")))
old_by_seed={int(r["training_seed"]):r for r in old}
def avg(metric): return float(np.mean([r[f"mean_{metric}"] for r in seed_means]))
mpc={m:float(np.mean([float(r[m]) for r in baseline if r["policy"]=="mpc"])) for m in metrics}
temp_changes=[r["mean_peak_temperature_C"]-float(old_by_seed[r["training_seed"]]["mean_peak_temperature_C"]) for r in seed_means]
fan_changes=[r["mean_fan_energy_Wh"]-float(old_by_seed[r["training_seed"]]["mean_fan_energy_Wh"]) for r in seed_means]
report=f"""# v05低频更新SAC pilot结论

## 训练协议

- 三个独立种子分别迁移actor/critic网络，但清空旧经验缓存；
- 每个种子训练5回合，每回合1200步；
- mini-batch 64，actor、critic和target每20个环境步更新一次；
- 总训练量18,000环境步；该pilot不是收敛性证明。

## 未见场景结果

- 严格任务通过：{sum(r['strict_feasible_scenarios'] for r in seed_means)}/30；
- SAC平均峰温：{avg('peak_temperature_C'):.2f} degC；
- SAC平均风机能耗：{avg('fan_energy_Wh'):.3f} Wh；
- MPC平均峰温：{mpc['peak_temperature_C']:.2f} degC；
- MPC平均风机能耗：{mpc['fan_energy_Wh']:.3f} Wh；
- SAC平均未满足能量：{avg('unmet_energy_kJ'):.3f} kJ；
- SAC平均安全层覆盖：{avg('safety_override_steps'):.2f}步/场景。

相对未进行v05训练的迁移策略，三个种子的峰温变化分别为{temp_changes[0]:+.2f}、{temp_changes[1]:+.2f}、{temp_changes[2]:+.2f} degC，风机能耗变化分别为{fan_changes[0]:+.3f}、{fan_changes[1]:+.3f}、{fan_changes[2]:+.3f} Wh。

seed11由节能型转向过度冷却，seed22的高温显著改善且仍比MPC节省风机能耗，seed33仍具有较大场景波动并依赖安全层。低频更新解决了计算性能问题，也证明seed22可以继续优化，但5回合不足以使三个种子收敛到一致策略。

## 论文边界

当前最稳健结论仍是：1.7 kW架构保证当前测试集任务可行；MPC提供最一致的温度控制；SAC能够探索不同温度--风机能耗折中，但短程训练的种子稳定性不足。下一轮应延长低频更新训练并设置基于验证集的模型选择规则，而不是按测试集挑选种子。
"""
(OUT/"v05低频更新SAC_pilot结论.md").write_text(report,encoding="utf-8")

try:
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,5),constrained_layout=True)
    mpc_rows=[r for r in baseline if r["policy"]=="mpc"]
    ax.scatter([float(r["fan_energy_Wh"]) for r in mpc_rows],[float(r["peak_temperature_C"]) for r in mpc_rows],label="MPC",alpha=.7)
    colors={11:"#7570b3",22:"#d95f02",33:"#e7298a"}
    for seed in seeds:
        rr=[r for r in all_sac if int(r["training_seed"])==seed]
        ax.scatter([float(r["fan_energy_Wh"]) for r in rr],[float(r["peak_temperature_C"]) for r in rr],label=f"SAC pilot {seed}",alpha=.7,c=colors[seed])
    ax.axhline(65,c="crimson",ls="--"); ax.set(xlabel="Fan energy (Wh)",ylabel="Peak stack temperature (degC)",title="v05 low-frequency-update SAC pilot")
    ax.grid(alpha=.25); ax.legend(frameon=False,fontsize=8); fig.savefig(OUT/"v05_pilot_pareto.png",dpi=220); plt.close(fig)
except ImportError: pass
print(report)
