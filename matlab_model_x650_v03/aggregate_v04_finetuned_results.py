from pathlib import Path
import csv
import math
import numpy as np

ROOT=Path(__file__).resolve().parent; OUT=ROOT/"results"
seeds=(11,22,33); all_sac=[]; baseline=None
for training_seed in seeds:
    path=OUT/f"sac_v04_ft_seed{training_seed}_test_comparison_v03.csv"
    rows=list(csv.DictReader(path.open(encoding="utf-8-sig")))
    if baseline is None: baseline=[r for r in rows if r["policy"] in ("fixed","pi","mpc")]
    selected=[r for r in rows if r["policy"].startswith("sac_")]
    assert len(selected)==10
    for r in selected:
        r=dict(r); r["training_seed"]=str(training_seed); all_sac.append(r)

metrics=("peak_temperature_C","total_reward","fan_energy_Wh","hydrogen_g",
         "unmet_energy_kJ","final_SOC","action_total_variation",
         "mission_energy_adequacy","equivalent_fully_served_time_min")

with (OUT/"v04_finetuned_all_sac_episodes.csv").open("w",newline="",encoding="utf-8-sig") as h:
    w=csv.DictWriter(h,fieldnames=all_sac[0].keys()); w.writeheader(); w.writerows(all_sac)

baseline_index={(r["policy"],int(float(r["seed"]))):r for r in baseline}
seed_rows=[]; paired_rows=[]
for training_seed in seeds:
    rr=[r for r in all_sac if int(r["training_seed"])==training_seed]
    record={"training_seed":training_seed}
    for metric in metrics:
        x=np.array([float(r[metric]) for r in rr]); record[f"mean_{metric}"]=float(x.mean())
        for base in ("mpc","pi"):
            delta=np.array([float(r[metric])-float(baseline_index[(base,int(float(r["seed"])))][metric]) for r in rr])
            ci=1.96*delta.std(ddof=1)/math.sqrt(len(delta))
            paired_rows.append({"training_seed":training_seed,"comparison":f"sac_minus_{base}",
                "metric":metric,"n":len(delta),"mean_delta":float(delta.mean()),
                "ci95_low":float(delta.mean()-ci),"ci95_high":float(delta.mean()+ci)})
    record["feasible_scenarios"]=sum(float(r["unmet_energy_kJ"])<.1 and float(r["final_SOC"])>.26 and float(r["violation_steps"])==0 for r in rr)
    seed_rows.append(record)

with (OUT/"v04_finetuned_seed_means.csv").open("w",newline="",encoding="utf-8-sig") as h:
    w=csv.DictWriter(h,fieldnames=seed_rows[0].keys()); w.writeheader(); w.writerows(seed_rows)
with (OUT/"v04_finetuned_paired_differences.csv").open("w",newline="",encoding="utf-8-sig") as h:
    w=csv.DictWriter(h,fieldnames=paired_rows[0].keys()); w.writeheader(); w.writerows(paired_rows)

tcrit=4.3026527299; inference=[]
for metric in metrics:
    x=np.array([r[f"mean_{metric}"] for r in seed_rows]); ci=tcrit*x.std(ddof=1)/math.sqrt(3)
    inference.append({"metric":metric,"training_seeds":3,"mean":float(x.mean()),
        "between_seed_std":float(x.std(ddof=1)),"ci95_low":float(x.mean()-ci),"ci95_high":float(x.mean()+ci)})
with (OUT/"v04_finetuned_training_seed_inference.csv").open("w",newline="",encoding="utf-8-sig") as h:
    w=csv.DictWriter(h,fieldnames=inference[0].keys()); w.writeheader(); w.writerows(inference)

def mean(metric): return next(r["mean"] for r in inference if r["metric"]==metric)
mpc_mean={m:float(np.mean([float(r[m]) for r in baseline if r["policy"]=="mpc"])) for m in metrics}
feasible_total=sum(r["feasible_scenarios"] for r in seed_rows)
baseline_feasible={policy:sum(float(r["unmet_energy_kJ"])<.1 and float(r["final_SOC"])>.26 and float(r["violation_steps"])==0
    for r in baseline if r["policy"]==policy) for policy in ("fixed","pi","mpc")}
seed_peak_deltas=[next(r["mean_delta"] for r in paired_rows if r["training_seed"]==seed and r["comparison"]=="sac_minus_mpc" and r["metric"]=="peak_temperature_C") for seed in seeds]
seed_fan_deltas=[next(r["mean_delta"] for r in paired_rows if r["training_seed"]==seed and r["comparison"]=="sac_minus_mpc" and r["metric"]=="fan_energy_Wh") for seed in seeds]
report=f"""# v04推荐架构SAC精调与配对测试结论

## 试验边界

- 架构：概念性1.5 kW PEMFC + 6S 5 Ah缓冲电池；
- 方法：分别从三个独立v03训练种子迁移，并在v04扩大扰动分布上精调10回合；
- 该结果属于迁移精调，不等同于v04从零训练；
- 每个训练种子使用10个相同的未见高空场景进行SAC、fixed、PI和MPC配对测试。

## 三训练种子总体结果

- SAC平均峰值温度：{mean('peak_temperature_C'):.2f} degC；
- SAC平均风机能耗：{mean('fan_energy_Wh'):.3f} Wh；
- SAC平均未满足能量：{mean('unmet_energy_kJ'):.2f} kJ；
- SAC平均任务能量满足率：{100*mean('mission_energy_adequacy'):.3f}%；
- SAC满足严格任务条件的回合：{feasible_total}/30；
- MPC平均峰值温度：{mpc_mean['peak_temperature_C']:.2f} degC；
- MPC平均风机能耗：{mpc_mean['fan_energy_Wh']:.3f} Wh；
- MPC平均任务能量满足率：{100*mpc_mean['mission_energy_adequacy']:.3f}%。
- fixed、PI和MPC严格任务通过数均为：{baseline_feasible['mpc']}/10；
- 三个SAC种子相对MPC的平均峰温差分别为：{seed_peak_deltas[0]:+.2f}、{seed_peak_deltas[1]:+.2f}、{seed_peak_deltas[2]:+.2f} degC；
- 三个SAC种子相对MPC的平均风机能耗差分别为：{seed_fan_deltas[0]:+.3f}、{seed_fan_deltas[1]:+.3f}、{seed_fan_deltas[2]:+.3f} Wh。

三种SAC训练种子的温度表现差异明显，说明10回合迁移精调尚未收敛到稳定一致的策略。当前结果可作为方法可运行性与种子敏感性证据，但不足以支持“SAC稳定优于MPC”的结论。

## 当前可以成立的结论

1. 1.5 kW + 5 Ah架构将旧架构约87%的能量满足率提高到严格扰动测试下约99%以上，但少数极端场景仍会出现缺能；
2. MPC保持最稳定的温度控制，平均峰值温度约{mpc_mean['peak_temperature_C']:.2f} degC；
3. SAC的风机能耗总体略低于MPC，但不同训练种子的温度和奖励差异较大；
4. 下一步需要延长精调或从零训练，并增加极端场景的能量安全约束，而不是挑选表现最好的随机种子。
"""
(OUT/"v04推荐架构SAC精调结论.md").write_text(report,encoding="utf-8")

try:
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,5),constrained_layout=True)
    mpc=[r for r in baseline if r["policy"]=="mpc"]
    ax.scatter([float(r["fan_energy_Wh"]) for r in mpc],[float(r["peak_temperature_C"]) for r in mpc],
               c="#1b9e77",alpha=.65,label="MPC scenarios")
    colors={11:"#7570b3",22:"#d95f02",33:"#e7298a"}
    for seed in seeds:
        rr=[r for r in all_sac if int(r["training_seed"])==seed]
        ax.scatter([float(r["fan_energy_Wh"]) for r in rr],[float(r["peak_temperature_C"]) for r in rr],
                   alpha=.65,label=f"SAC seed {seed}",c=colors[seed])
    ax.axhline(65,color="crimson",linestyle="--",label="temperature limit")
    ax.set(xlabel="Fan energy (Wh)",ylabel="Peak stack temperature (degC)",
           title="v04 matched unseen-scenario controller outcomes")
    ax.grid(alpha=.25); ax.legend(frameon=False,fontsize=8)
    fig.savefig(OUT/"v04_sac_mpc_pareto.png",dpi=220); plt.close(fig)
except ImportError: pass
print(report)
