from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'results'/'probability_dataset_v01'
OUT=ROOT/'results'/'input_ablation_v01'
df=pd.read_csv(DATA/'thermal_trajectories_v01.csv',usecols=[
    'scenario_id','split','controller','time_s','stack_temperature_C','fan_command',
    'altitude_m','ambient_temperature_C'])
rows=[]
for sid,g in df.groupby('scenario_id',sort=False):
    g=g.sort_values('time_s').reset_index(drop=True); h=20
    current=g.stack_temperature_C.iloc[:-h].to_numpy(); future=g.stack_temperature_C.iloc[h:].to_numpy()
    delta=future-current
    rows.append({'scenario_id':int(sid),'split':g.split.iloc[0],'controller':g.controller.iloc[0],
                 'altitude_m':g.altitude_m.iloc[0],'ambient_temperature_C':g.ambient_temperature_C.iloc[0],
                 'persistence_mae_10s_C':float(np.mean(np.abs(delta))),
                 'mean_abs_delta_10s_C':float(np.mean(np.abs(delta))),
                 'p95_abs_delta_10s_C':float(np.quantile(np.abs(delta),.95)),
                 'temperature_min_C':float(g.stack_temperature_C.min()),
                 'temperature_max_C':float(g.stack_temperature_C.max()),
                 'overtemp_points':int((g.stack_temperature_C>65).sum())})
audit=pd.DataFrame(rows)
audit.to_csv(OUT/'ood_shortcut_scenario_audit_v01.csv',index=False)
summary=audit.groupby(['split','controller']).agg(
    n_scenarios=('scenario_id','count'),persistence_mae_mean_C=('persistence_mae_10s_C','mean'),
    p95_delta_mean_C=('p95_abs_delta_10s_C','mean'),overtemp_points=('overtemp_points','sum')).reset_index()
summary.to_csv(OUT/'ood_shortcut_summary_v01.csv',index=False)
meta={'horizon_s':10,'rows':len(df),'scenarios':len(audit),
      'interpretation':'Persistence error measures intrinsic short-horizon predictability, not learned-model skill.'}
(OUT/'ood_shortcut_audit_v01.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
print(summary.to_string(index=False))
