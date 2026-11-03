from pathlib import Path
import csv
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
rows = list(csv.DictReader((OUT / "mass_capacity_sensitivity_v03.csv").open(encoding="utf-8-sig")))

def f(row, key): return float(row[key])
def truth(value): return str(value).strip().lower() in {"1", "true", "yes"}

groups = defaultdict(list)
for row in rows:
    groups[(int(f(row, "fc_rating_W")), f(row, "battery_capacity_Ah"))].append(row)

summary=[]
for (rating, ah), rr in sorted(groups.items()):
    feasible=sum(truth(r["mission_feasible"]) for r in rr)
    valid=[r for r in rr if np.isfinite(f(r,"final_SOC"))]
    summary.append({"fc_rating_W":rating,"battery_capacity_Ah":ah,"sensitivity_cases":len(rr),
                    "feasible_cases":feasible,"feasible_fraction":feasible/len(rr),
                    "minimum_final_SOC":min(f(r,"final_SOC") for r in valid),
                    "maximum_unmet_energy_kJ":max(f(r,"unmet_energy_kJ") for r in valid),
                    "maximum_takeoff_mass_kg":max(f(r,"takeoff_mass_kg") for r in valid)})

with (OUT/"mass_capacity_sensitivity_summary_v03.csv").open("w",newline="",encoding="utf-8-sig") as h:
    w=csv.DictWriter(h,fieldnames=summary[0].keys()); w.writeheader(); w.writerows(summary)

try:
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(1,2,figsize=(10.5,4.5),constrained_layout=True)
    labels=[f"{r['fc_rating_W']/1000:.1f} kW\n{r['battery_capacity_Ah']:g} Ah" for r in summary]
    axes[0].bar(labels,[100*r["feasible_fraction"] for r in summary],color="#1b9e77")
    axes[0].set_ylabel("Feasible sensitivity cases (%)"); axes[0].set_ylim(0,105)
    axes[0].set_title("Robustness across 9 mass assumptions"); axes[0].grid(axis="y",alpha=.25)
    for i,r in enumerate(summary): axes[0].text(i,100*r["feasible_fraction"]+2,f"{r['feasible_cases']}/9",ha="center",fontsize=9)
    axes[1].bar(labels,[r["maximum_unmet_energy_kJ"] for r in summary],color="#d95f02")
    axes[1].set_ylabel("Worst unmet energy (kJ)"); axes[1].set_title("Worst-case energy shortfall")
    axes[1].grid(axis="y",alpha=.25)
    fig.savefig(OUT/"mass_capacity_sensitivity_v03.png",dpi=220); plt.close(fig)
except ImportError:
    pass

print(summary)
