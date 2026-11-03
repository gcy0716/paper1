from pathlib import Path
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parent; BASE=ROOT/'results'/'input_ablation_replication_v01'
d=pd.read_csv(BASE/'input_replication_predictions_v01.csv')
d['abs_error_C']=(d.actual_temperature_C-d.p50_C).abs(); d['covered']=(d.actual_temperature_C>=d.p10_C)&(d.actual_temperature_C<=d.p90_C)
d['event63']=d.actual_temperature_C>63; d['alarm63']=d.p90_C>63
rows=[]
for keys,g in d.groupby(['training_seed','variant','split','controller']):
    event=g.event63; tp=(event&g.alarm63).sum()
    rows.append(dict(zip(['training_seed','variant','split','controller'],keys))|{
        'n_samples':len(g),'mae_C':g.abs_error_C.mean(),'coverage':g.covered.mean(),
        'event63_count':int(event.sum()),'recall63':tp/max(int(event.sum()),1)})
m=pd.DataFrame(rows); m.to_csv(BASE/'input_replication_controller_metrics_v01.csv',index=False)
s=m.groupby(['variant','split','controller']).agg(n_seeds=('training_seed','count'),mean_mae_C=('mae_C','mean'),std_mae_C=('mae_C','std'),mean_coverage=('coverage','mean'),std_coverage=('coverage','std'),event63_count_per_seed=('event63_count','mean'),mean_recall63=('recall63','mean')).reset_index()
s.to_csv(BASE/'input_replication_controller_summary_v01.csv',index=False)
print(s.to_string(index=False))
