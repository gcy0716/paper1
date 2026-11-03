from pathlib import Path
import csv, math
import numpy as np

ROOT=Path(__file__).resolve().parent; OUT=ROOT/"results"; seeds=(11,22,33)
all_sac=[]; baseline=None
for seed in seeds:
    rows=list(csv.DictReader((OUT/f"sac_v05_transfer_seed{seed}_test_comparison_v03.csv").open(encoding="utf-8-sig")))
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
        for base in ("mpc","pi"):
            d=np.array([float(r[metric])-float(base_index[(base,int(float(r["seed"])))][metric]) for r in rr])
            ci=1.96*d.std(ddof=1)/math.sqrt(len(d))
            paired.append({"training_seed":seed,"comparison":f"sac_minus_{base}","metric":metric,
                "n":len(d),"mean_delta":float(d.mean()),"ci95_low":float(d.mean()-ci),"ci95_high":float(d.mean()+ci)})
    rec["strict_feasible_scenarios"]=sum(float(r["unmet_energy_kJ"])<.1 and float(r["final_SOC"])>.26 and float(r["violation_steps"])==0 for r in rr)
    seed_means.append(rec)

for name,data in (("v05_transfer_all_sac_episodes.csv",all_sac),("v05_transfer_seed_means.csv",seed_means),
                  ("v05_transfer_paired_differences.csv",paired)):
    with (OUT/name).open("w",newline="",encoding="utf-8-sig") as h:
        w=csv.DictWriter(h,fieldnames=data[0].keys()); w.writeheader(); w.writerows(data)

def sac_mean(metric): return float(np.mean([r[f"mean_{metric}"] for r in seed_means]))
mpc={m:float(np.mean([float(r[m]) for r in baseline if r["policy"]=="mpc"])) for m in metrics}
peak_delta=[r["mean_peak_temperature_C"]-mpc["peak_temperature_C"] for r in seed_means]
fan_delta=[r["mean_fan_energy_Wh"]-mpc["fan_energy_Wh"] for r in seed_means]
fan_saving=100*(mpc["fan_energy_Wh"]-sac_mean("fan_energy_Wh"))/mpc["fan_energy_Wh"]
report=f"""# v05稳健功率裕度与SAC迁移评估

## 架构边界

- 1.6 kW仍不能通过最坏场景；
- 1.7 kW + 5 Ah在原10个严格未见场景中使fixed、安全PI和MPC全部零缺能；
- 1.8 kW虽然提高SOC，但固定风机控制峰温达到65.63 degC，因此不是最小或普遍更优方案；
- 1.7 kW概念起飞质量为5.745 kg，仍低于6.3 kg限制。

## 三种子迁移测试

三个v04 SAC策略未经v05完整重训，直接在1.7 kW架构上关闭探索噪声测试：

- 严格任务通过：{sum(r['strict_feasible_scenarios'] for r in seed_means)}/30；
- 平均峰值温度：{sac_mean('peak_temperature_C'):.2f} degC；
- 平均风机能耗：{sac_mean('fan_energy_Wh'):.3f} Wh；
- 平均氢耗：{sac_mean('hydrogen_g'):.2f} g；
- 未满足能量：{sac_mean('unmet_energy_kJ'):.3f} kJ；
- 温度越界步数：{sac_mean('violation_steps'):.1f}；
- 平均安全层覆盖次数：{sac_mean('safety_override_steps'):.1f}步/场景。

对应MPC平均峰温为{mpc['peak_temperature_C']:.2f} degC，风机能耗为{mpc['fan_energy_Wh']:.3f} Wh。SAC平均节省风机能耗约{fan_saving:.1f}%，但三个种子相对MPC的峰温差分别为{peak_delta[0]:+.2f}、{peak_delta[1]:+.2f}、{peak_delta[2]:+.2f} degC；风机能耗差分别为{fan_delta[0]:+.3f}、{fan_delta[1]:+.3f}、{fan_delta[2]:+.3f} Wh。

## 可写入论文的结论边界

1. 将额定功率提高到1.7 kW消除了当前严格测试集中的架构级缺能，说明功率裕度而非风机控制是任务可行性的主导因素；
2. 未重训SAC仍能保持零温度越界和零缺能，并平均降低风机能耗，但温度表现存在明显种子差异；
3. seed33触发了安全层覆盖，证明独立安全过滤层在分布迁移时仍有实际作用；
4. 这些结果属于迁移评估，不得写成v05 SAC已经收敛或全面优于MPC；完整结论仍需从零训练或清空经验缓存后的长程训练。
"""
(OUT/"v05稳健架构与迁移评估结论.md").write_text(report,encoding="utf-8")

try:
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,5),constrained_layout=True)
    mpc_rows=[r for r in baseline if r["policy"]=="mpc"]
    ax.scatter([float(r["fan_energy_Wh"]) for r in mpc_rows],[float(r["peak_temperature_C"]) for r in mpc_rows],label="MPC",alpha=.7)
    colors={11:"#7570b3",22:"#d95f02",33:"#e7298a"}
    for seed in seeds:
        rr=[r for r in all_sac if int(r["training_seed"])==seed]
        ax.scatter([float(r["fan_energy_Wh"]) for r in rr],[float(r["peak_temperature_C"]) for r in rr],label=f"SAC transfer {seed}",alpha=.7,c=colors[seed])
    ax.axhline(65,c="crimson",ls="--",label="temperature limit")
    ax.set(xlabel="Fan energy (Wh)",ylabel="Peak stack temperature (degC)",title="v05 robust architecture: transfer-only evaluation")
    ax.grid(alpha=.25); ax.legend(frameon=False,fontsize=8); fig.savefig(OUT/"v05_transfer_pareto.png",dpi=220); plt.close(fig)
except ImportError: pass
print(report)
