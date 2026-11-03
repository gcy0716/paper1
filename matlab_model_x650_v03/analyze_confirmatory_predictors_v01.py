from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parent; OUT=ROOT/'results'/'confirmatory_prediction_v01'
d=pd.read_csv(OUT/'confirmatory_predictor_outputs_v01.csv'); d['ae']=(d.actual_temperature_C-d.p50_C).abs(); d['covered']=(d.actual_temperature_C>=d.p10_C)&(d.actual_temperature_C<=d.p90_C); d['event']=d.actual_temperature_C>=63; d['alarm']=d.p90_C>=63
rows=[]
for keys,g in d.groupby(['training_seed','variant','controller']):
 tp=(g.event&g.alarm).sum(); fp=(~g.event&g.alarm).sum(); fn=(g.event&~g.alarm).sum()
 rows.append(dict(zip(['training_seed','variant','controller'],keys))|{'n_windows':len(g),'risk_windows':int(g.event.sum()),'mae_C':g.ae.mean(),'coverage':g.covered.mean(),'recall63':tp/max(tp+fn,1),'precision63':tp/max(tp+fp,1)})
m=pd.DataFrame(rows); m.to_csv(OUT/'confirmatory_controller_metrics_v01.csv',index=False)
s=m.groupby(['variant','controller']).agg(n_seeds=('training_seed','count'),n_windows_per_seed=('n_windows','mean'),risk_windows_per_seed=('risk_windows','mean'),mean_mae_C=('mae_C','mean'),std_mae_C=('mae_C','std'),mean_coverage=('coverage','mean'),std_coverage=('coverage','std'),mean_recall63=('recall63','mean'),std_recall63=('recall63','std'),mean_precision63=('precision63','mean')).reset_index(); s.to_csv(OUT/'confirmatory_controller_summary_v01.csv',index=False)
macro=m.groupby(['training_seed','variant'])[['mae_C','coverage','recall63','precision63']].mean().reset_index(); ms=macro.groupby('variant').agg(['mean','std']); ms.to_csv(OUT/'confirmatory_macro_summary_v01.csv')
audit={'prediction_rows':len(d),'temperature_window_C':[61,67],'risk_threshold_C':63,'training_seeds':sorted(d.training_seed.unique().tolist()),'models_frozen':True,'calibration_reused':True,'old_final_test_used':False}
(OUT/'confirmatory_prediction_audit_v01.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(s.to_string(index=False)); print('\nMACRO\n',ms.to_string())
