from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'/'input_ablation_v01'
BASE=ROOT/'results'/'probability_dataset_v01'/'quantile_gru_results'
d=pd.read_csv(OUT/'input_ablation_metrics_v01.csv')
# Source column is called variant in the baseline file; restore it before assigning model label.
full0=pd.read_csv(BASE/'quantile_gru_calibrated_metrics_v01.csv')
full0=full0[full0.horizon_s==10].rename(columns={'variant':'calibration_variant'})
full0['variant']='full'; full0['n_features']=19; full0['calibration_expansion_C']=0.0381851196289062
full0['crossing_rate']=0.0
cols=['variant','split','calibration_variant','n_features','calibration_expansion_C','mae_C','rmse_C','picp_p10_p90','mpiw_p10_p90_C','crossing_rate','overtemp_events','overtemp_recall','overtemp_precision']
combined=pd.concat([full0[cols],d[cols]],ignore_index=True)
combined.to_csv(OUT/'input_ablation_combined_metrics_v01.csv',index=False)

plot=combined[(combined.split.isin(['test','ood_test'])) & (combined.calibration_variant=='validation_conformal')]
order=['full','no_task','no_environment','no_thermal_history','minimal']
labels=['Full','No task','No environment','Current thermal only','Minimal']
fig,axes=plt.subplots(1,3,figsize=(7.2,3.0))
metrics=[('mae_C','MAE (°C)'),('picp_p10_p90','P10–P90 coverage (%)'),('overtemp_recall','Overtemperature recall (%)')]
colors={'test':'#0077BB','ood_test':'#EE7733'}
for ax,(metric,ylabel) in zip(axes,metrics):
    for i,split in enumerate(['test','ood_test']):
        z=plot[plot.split==split].set_index('variant').reindex(order)
        values=z[metric].to_numpy()*(100 if metric!='mae_C' else 1)
        x=[j+(i-.5)*.18 for j in range(len(order))]
        ax.bar(x,values,width=.18,color=colors[split],label='Test' if split=='test' else 'OOD test')
    ax.set_xticks(range(len(order)),labels,rotation=35,ha='right'); ax.set_ylabel(ylabel); ax.grid(axis='y',alpha=.2)
axes[1].axhline(80,color='#555555',ls='--',lw=1)
axes[0].legend(frameon=False,fontsize=7)
fig.suptitle('Input-source ablation for the 10 s quantile GRU',fontsize=10)
fig.tight_layout()
for ext in ['png','pdf']:
    fig.savefig(OUT/f'Fig_input_ablation_v01.{ext}',dpi=600,bbox_inches='tight')
plt.close(fig)

audit={'new_models':4,'full_model_reused':True,'horizon_s':10,'splits':['validation','test','ood_test'],
       'calibration_source':'validation only','final_test_used':False,'combined_metric_rows':len(combined)}
(OUT/'input_ablation_audit_v01.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(plot[['variant','split','mae_C','picp_p10_p90','mpiw_p10_p90_C','overtemp_recall']].to_string(index=False))
