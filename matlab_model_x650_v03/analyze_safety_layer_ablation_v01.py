from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent; OUT=ROOT/'results'/'safety_layer_ablation_v01'
d=pd.read_csv(OUT/'safety_layer_ablation_episodes_v01.csv'); rng=np.random.default_rng(20260911); B=10000
feasible=d.groupby('scenario_seed').unmet_energy_kJ.max(); feasible=feasible[feasible<.1].index
rows=[]
for scope,z in [('all',d),('architecture_feasible',d[d.scenario_seed.isin(feasible)])]:
  for seed in [11,22,33]:
    q=z[z.agent_seed==seed]
    for metric in ['peak_temperature_C','fan_energy_Wh','unmet_energy_kJ']:
      a=q[q.variant=='reactive'].set_index('scenario_seed')[metric]; b=q[q.variant=='p90_reactive'].set_index('scenario_seed')[metric]
      x=(b-a).dropna().to_numpy(); bs=x[rng.integers(0,len(x),(B,len(x)))].mean(1)
      rows.append({'scope':scope,'agent_seed':seed,'metric':metric,'n_scenarios':len(x),'mean_delta_p90_minus_reactive':x.mean(),'ci95_low':np.quantile(bs,.025),'ci95_high':np.quantile(bs,.975)})
pd.DataFrame(rows).to_csv(OUT/'safety_layer_ablation_bootstrap_v01.csv',index=False)
d.groupby('variant').agg(episodes=('variant','size'),violated_episodes=('violated_limit','sum'),terminated_over_70C=('terminated_over_70C','sum'),violation_steps=('violation_steps','sum'),mean_peak_temperature_C=('peak_temperature_C','mean'),mean_fan_energy_Wh=('fan_energy_Wh','mean')).to_csv(OUT/'safety_layer_ablation_overall_v01.csv')
fig,ax=plt.subplots(figsize=(5.5,3.5)); s=d.groupby('variant').agg(violated=('violated_limit','sum'),safe=('violated_limit',lambda x:len(x)-x.sum())).reindex(['none','reactive','p90_reactive'])
ax.bar(['No safety layer','Reactive','P90 + reactive'],s.violated,color='#CC3311',label='Violated 65 °C'); ax.bar(['No safety layer','Reactive','P90 + reactive'],s.safe,bottom=s.violated,color='#009988',label='No violation')
ax.set_ylabel('Episodes'); ax.set_title('Safety-layer necessity across 60 episodes per variant'); ax.legend(frameon=False); ax.spines[['top','right']].set_visible(False); ax.grid(axis='y',alpha=.2)
fig.tight_layout();
for ext in ['png','pdf']: fig.savefig(OUT/f'Fig_safety_layer_necessity_v01.{ext}',dpi=600,bbox_inches='tight')
plt.close(fig)
audit={'rows':len(d),'expected_rows':180,'violation_steps':int(d.violation_steps.sum()),'feasible_scenarios':len(feasible),'final_test_used':False}
(OUT/'safety_layer_ablation_audit_v01.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(json.dumps(audit)); print(pd.read_csv(OUT/'safety_layer_ablation_overall_v01.csv').to_string(index=False))
