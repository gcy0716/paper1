"""Verification and exploratory analyses with explicit sampling units."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.linalg import expm

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'
ROOT = HERE.parents[1]


def benchmark():
    old = pd.read_csv(ROOT / 'matlab_model_x650_v03/results/reviewer_alarm_ablation_v01/episodes_v01.csv')
    new = pd.read_csv(OUT / 'benchmark_reproduction.csv')
    ref = old[(old.scenario_seed == 4301) & (old.agent_seed == 11) & (old.variant == 'alarm63')].iloc[0]
    columns = ['peak_temperature_C','fan_energy_Wh','hydrogen_g','final_SOC','unmet_energy_kJ',
               'violation_steps','predictive_override_steps','reactive_override_steps']
    error = {c:float(abs(new.iloc[0][c] - ref[c])) for c in columns}
    assert max(error.values()) < 1e-8, error
    (OUT / 'benchmark_audit.json').write_text(json.dumps(error, indent=2))
    return error


def thermal():
    rows = []
    for seed in range(4301,4321):
        trace = pd.read_csv(OUT / f'thermal_trace_{seed}.csv')
        p = loadmat(OUT / f'thermal_parameters_{seed}.mat', simplify_cells=True)['p']
        v = p['thermal']; initial = np.array([v['Tstack0_C'],v['Tair0_C']])
        states = {n:initial.copy() for n in [1,2,5,10]}
        exact = initial.copy()
        all_states = {n:[] for n in states}; exact_states=[]
        for r in trace.itertuples(index=False):
            cs = v['Cstack_JpK'] * r.capScale; ca=v['Cair_JpK']
            g = r.uaScale*r.fan**.7*(r.rho/p['env']['rho0'])**.5
            sa=v['UA_stack_air_max_WpK']*g; aa=v['UA_air_amb_max_WpK']*g
            pas=v['UA_passive_WpK']
            a=np.array([[-(sa+pas)/cs,sa/cs],[sa/ca,-(sa+aa)/ca]])
            b=np.array([(r.Qgen_W+pas*r.Tamb_C)/cs,aa*r.Tamb_C/ca])
            aug=np.zeros((3,3)); aug[:2,:2]=a; aug[:2,2]=b
            exact=(expm(aug*.5) @ np.r_[exact,1])[:2]; exact_states.append(exact.copy())
            for n in states:
                for _ in range(n): states[n] += (.5/n)*(a@states[n]+b)
                all_states[n].append(states[n].copy())
        exact_states=np.array(exact_states)
        for n,values in all_states.items():
            values=np.array(values)
            replay_error=float(np.max(np.abs(values-trace[['Ts_C','Ta_C']].to_numpy()))) if n==1 else np.nan
            if n==1: assert replay_error < 1e-8, (seed,replay_error)
            rows.append(dict(scenario_seed=seed,step_s=.5/n,n_samples=len(trace),
                replay_error_C=replay_error,
                max_stack_error_C=float(np.abs(values[:,0]-exact_states[:,0]).max()),
                max_air_error_C=float(np.abs(values[:,1]-exact_states[:,1]).max()),
                peak_stack_C=float(values[:,0].max()),exact_peak_stack_C=float(exact_states[:,0].max()),
                peak_difference_C=float(values[:,0].max()-exact_states[:,0].max())))
        print('THERMAL_REPLAY',seed,flush=True)
    df=pd.DataFrame(rows); df.to_csv(OUT/'thermal_replay_convergence.csv',index=False)
    summary=df.groupby('step_s').agg(max_stack_error_C=('max_stack_error_C','max'),
        mean_max_stack_error_C=('max_stack_error_C','mean'),
        max_abs_peak_difference_C=('peak_difference_C',lambda x: x.abs().max()))
    summary.to_csv(OUT/'thermal_replay_summary.csv')
    sensitivity=pd.read_csv(OUT/'thermal_sensitivity.csv')
    assert len(sensitivity)==220 and not sensitivity.duplicated(['scenario_seed','variant']).any()
    base=sensitivity[sensitivity.variant=='nominal'].set_index('scenario_seed')
    for metric in ['peak_temperature_C','fan_energy_Wh','hydrogen_g','final_SOC','unmet_energy_kJ']:
        sensitivity['delta_'+metric]=sensitivity[metric]-sensitivity.scenario_seed.map(base[metric])
    sensitivity.to_csv(OUT/'thermal_sensitivity_paired.csv',index=False)
    s=sensitivity.groupby('variant').agg(episodes=('scenario_seed','size'),
        mean_peak_C=('peak_temperature_C','mean'),max_peak_C=('peak_temperature_C','max'),
        mean_delta_peak_C=('delta_peak_temperature_C','mean'),
        min_delta_peak_C=('delta_peak_temperature_C','min'),max_delta_peak_C=('delta_peak_temperature_C','max'),
        mean_fan_Wh=('fan_energy_Wh','mean'),mean_delta_fan_Wh=('delta_fan_energy_Wh','mean'),
        violated_episodes=('violation_steps',lambda x:int((x>0).sum())),
        terminated_episodes=('duration_s',lambda x:int((x<1200).sum())))
    s.to_csv(OUT/'thermal_sensitivity_summary.csv'); print(summary); print(s)


def select_alarm():
    df=pd.read_csv(OUT/'development_episodes.csv')
    assert len(df)==300 and df.scenario_seed.nunique()==10
    assert not df.duplicated(['scenario_seed','agent_seed','variant']).any()
    assert (df.duration_s==1200).all(), 'Truncated episodes cannot be matched using mission fan energy'
    summary=df.groupby('variant').agg(mean_fan_Wh=('fan_energy_Wh','mean'),
        mean_peak_C=('peak_temperature_C','mean'),threshold_C=('threshold_C','first'),
        violated_episodes=('violation_steps',lambda x:int((x>0).sum())),
        max_peak_C=('peak_temperature_C','max'))
    target=summary.loc['calibrated_p90','mean_fan_Wh']
    candidates=summary[np.isfinite(summary.threshold_C)].copy()
    candidates['absolute_energy_mismatch_Wh']=(candidates.mean_fan_Wh-target).abs()
    chosen=candidates.sort_values(['absolute_energy_mismatch_Wh','threshold_C']).head(1)
    chosen['p90_target_fan_Wh']=target
    chosen.to_csv(OUT/'selected_alarm.csv',index=False); summary.to_csv(OUT/'development_summary.csv')
    print(chosen)


def refinement():
    df=pd.read_csv(OUT/'closed_loop_thermal_refinement.csv')
    assert len(df)==80
    base=pd.read_csv(OUT/'thermal_sensitivity.csv').query("variant=='nominal'").set_index('scenario_seed')
    same=df.query('thermal_step_s==0.5').set_index('scenario_seed')
    metrics=['peak_temperature_C','fan_energy_Wh','hydrogen_g','final_SOC','unmet_energy_kJ','violation_steps','reactive_override_steps']
    error=(same[metrics]-base[metrics]).abs().to_numpy().max()
    assert error<1e-8,error
    fine=df.query('thermal_step_s==0.05').set_index('scenario_seed')
    for metric in metrics:
        df['delta_'+metric]=df[metric]-df.scenario_seed.map(fine[metric])
    df.to_csv(OUT/'closed_loop_thermal_refinement_paired.csv',index=False)
    summary=df.groupby('thermal_step_s').agg(max_abs_peak_delta_C=('delta_peak_temperature_C',lambda x:x.abs().max()),
        mean_peak_delta_C=('delta_peak_temperature_C','mean'),
        max_abs_fan_delta_Wh=('delta_fan_energy_Wh',lambda x:x.abs().max()),
        violated_episodes=('violation_steps',lambda x:int((x>0).sum())))
    summary.to_csv(OUT/'closed_loop_thermal_refinement_summary.csv')
    print('Substep=1 reproduction max error',error); print(summary)


def holdout():
    df=pd.read_csv(OUT/'holdout_episodes.csv')
    chosen=pd.read_csv(OUT/'selected_alarm.csv').threshold_C.iloc[0]
    variants=df.variant.unique(); expected=60*len(variants)
    assert len(df)==expected and set(df.scenario_seed)==set(range(4501,4521))
    assert not df.duplicated(['scenario_seed','agent_seed','variant']).any()
    assert (df.duration_s==1200).all(), 'Report censoring before interpreting mission energy'
    summary=df.groupby('variant').agg(n=('scenario_seed','size'),
        mean_peak_C=('peak_temperature_C','mean'),mean_fan_Wh=('fan_energy_Wh','mean'),
        mean_hydrogen_g=('hydrogen_g','mean'),mean_final_SOC=('final_SOC','mean'),
        mean_unmet_kJ=('unmet_energy_kJ','mean'),violated_episodes=('violation_steps',lambda x:int((x>0).sum())),
        terminated_episodes=('duration_s',lambda x:int((x<1200).sum())))
    summary.to_csv(OUT/'holdout_summary.csv')
    metrics=['peak_temperature_C','fan_energy_Wh','hydrogen_g','final_SOC','unmet_energy_kJ']
    cluster=df.groupby(['scenario_seed','variant'])[metrics].mean()
    rng=np.random.default_rng(20261104); draws=rng.integers(0,20,(10000,20)); rows=[]
    for variant in variants:
        if variant=='calibrated_p90':continue
        for metric in metrics:
            table=cluster[metric].unstack('variant').sort_index()
            delta=(table.calibrated_p90-table[variant]).to_numpy()
            lo,hi=np.quantile(delta[draws].mean(axis=1),[.025,.975])
            rows.append(dict(comparator=variant,metric=metric,mean_difference=delta.mean(),low95=lo,high95=hi))
    pd.DataFrame(rows).to_csv(OUT/'holdout_paired_bootstrap.csv',index=False)
    print('Selected threshold',chosen); print(summary); print(pd.DataFrame(rows).head(10))


def refined_holdout():
    fine=pd.read_csv(OUT/'refined_holdout_episodes.csv')
    coarse=pd.read_csv(OUT/'holdout_episodes.csv')
    assert len(fine)==120 and set(fine.scenario_seed)==set(range(4501,4521))
    assert not fine.duplicated(['scenario_seed','agent_seed','variant']).any()
    assert (fine.duration_s==1200).all()
    metrics=['peak_temperature_C','fan_energy_Wh','hydrogen_g','final_SOC','unmet_energy_kJ']
    fine.groupby('variant').agg(n=('scenario_seed','size'),mean_peak_C=('peak_temperature_C','mean'),
        mean_fan_Wh=('fan_energy_Wh','mean'),mean_hydrogen_g=('hydrogen_g','mean'),
        mean_final_SOC=('final_SOC','mean'),mean_unmet_kJ=('unmet_energy_kJ','mean'),
        violated_episodes=('violation_steps',lambda x:int((x>0).sum()))).to_csv(OUT/'refined_holdout_summary.csv')
    comparator=next(v for v in fine.variant.unique() if v!='calibrated_p90')
    draws=np.random.default_rng(20261104).integers(0,20,(10000,20)); rows=[]
    for metric in metrics:
        deltas={}
        for label,d in [('coarse',coarse),('fine',fine)]:
            p=d.groupby(['scenario_seed','variant'])[metric].mean().unstack().sort_index()
            deltas[label]=(p.calibrated_p90-p[comparator]).to_numpy()
        for label,x in [('coarse',deltas['coarse']),('fine',deltas['fine']),('fine_minus_coarse',deltas['fine']-deltas['coarse'])]:
            lo,hi=np.quantile(x[draws].mean(axis=1),[.025,.975])
            rows.append(dict(comparator=comparator,metric=metric,thermal_integration=label,
                mean_difference=x.mean(),low95=lo,high95=hi,max_abs_scenario_difference=np.abs(x).max()))
    pd.DataFrame(rows).to_csv(OUT/'refined_holdout_paired_bootstrap.csv',index=False)
    paired=fine.merge(coarse,on=['scenario_seed','agent_seed','variant'],suffixes=('_fine','_coarse'),validate='one_to_one')
    for m in metrics:paired['delta_'+m]=paired[m+'_fine']-paired[m+'_coarse']
    paired.to_csv(OUT/'refined_holdout_episode_changes.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':
    import sys
    {'benchmark':benchmark,'thermal':thermal,'refinement':refinement,'select':select_alarm,'holdout':holdout,'refined_holdout':refined_holdout}[sys.argv[1]]()
