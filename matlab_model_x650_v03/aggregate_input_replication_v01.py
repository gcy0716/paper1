from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'results'/'input_ablation_replication_v01'
frames=[]
for seed in [20260911,20260912,20260913]:
    p=BASE/f'seed_{seed}'/'input_ablation_metrics_v01.csv'
    d=pd.read_csv(p); d['training_seed']=seed; frames.append(d)
allm=pd.concat(frames,ignore_index=True)
allm.to_csv(BASE/'input_replication_all_metrics_v01.csv',index=False)
z=allm[(allm.calibration_variant=='validation_conformal') & allm.split.isin(['test','ood_test'])]
summary=z.groupby(['variant','split']).agg(
    n_seeds=('training_seed','count'),mean_mae_C=('mae_C','mean'),std_mae_C=('mae_C','std'),
    mean_coverage=('picp_p10_p90','mean'),std_coverage=('picp_p10_p90','std'),
    mean_interval_width_C=('mpiw_p10_p90_C','mean'),
    mean_overtemp_recall=('overtemp_recall','mean'),std_overtemp_recall=('overtemp_recall','std')).reset_index()
summary.to_csv(BASE/'input_replication_summary_v01.csv',index=False)

fig,axes=plt.subplots(1,3,figsize=(7.2,3.0))
for ax,(metric,label,scale) in zip(axes,[('mae_C','MAE (°C)',1),('picp_p10_p90','Coverage (%)',100),('overtemp_recall','Overtemperature recall (%)',100)]):
    for i,variant in enumerate(['full','minimal']):
        for j,split in enumerate(['test','ood_test']):
            q=z[(z.variant==variant)&(z.split==split)]
            x=i+(j-.5)*.18
            ax.scatter([x]*len(q),q[metric]*scale,color=['#0077BB','#EE7733'][j],s=25,alpha=.8)
            ax.errorbar(x,q[metric].mean()*scale,yerr=q[metric].std(ddof=1)*scale,fmt='_',color='black',capsize=3)
    ax.set_xticks([0,1],['Full','Minimal']); ax.set_ylabel(label); ax.grid(axis='y',alpha=.2)
axes[1].axhline(80,color='#555555',ls='--',lw=1)
axes[0].scatter([],[],color='#0077BB',label='Test'); axes[0].scatter([],[],color='#EE7733',label='OOD test'); axes[0].legend(frameon=False,fontsize=7)
fig.suptitle('Three-seed replication of input-source ablation',fontsize=10); fig.tight_layout()
for ext in ['png','pdf']: fig.savefig(BASE/f'Fig_input_replication_v01.{ext}',dpi=600,bbox_inches='tight')
plt.close(fig)
audit={'training_seeds':[20260911,20260912,20260913],'models':6,'variants':['full','minimal'],'final_test_used':False}
(BASE/'input_replication_audit_v01.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(summary.to_string(index=False))
