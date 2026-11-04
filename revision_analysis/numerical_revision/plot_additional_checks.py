"""Python-only publication figures from the retained revision CSV results."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent; DATA=HERE/'results'; OUT=HERE/'figures'
OUT.mkdir(exist_ok=True)
BLUE,TEAL,GOLD,RED='#365E87','#25877F','#B3863F','#B85B4B'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':7.4,'axes.labelsize':7.7,'axes.titlesize':8.5,'svg.fonttype':'none',
    'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,
    'axes.linewidth':.7,'legend.frameon':False,'figure.facecolor':'white'})

def export(fig,name):
    fig.canvas.draw()
    for ext in ['svg','pdf','png','tiff']:
        kw={'dpi':600}
        if ext=='tiff':kw['pil_kwargs']={'compression':'tiff_lzw'}
        fig.savefig(OUT/f'{name}.{ext}',**kw)
    plt.close(fig)

def thermal():
    r=pd.read_csv(DATA/'thermal_replay_summary.csv').sort_values('step_s')
    c=pd.read_csv(DATA/'closed_loop_thermal_refinement_summary.csv')
    fig=plt.figure(figsize=(183/25.4,145/25.4)); gs=fig.add_gridspec(2,2,height_ratios=[1,1.9],
        left=.17,right=.97,bottom=.10,top=.94,hspace=.55,wspace=.42)
    a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[0,1]);ax=fig.add_subplot(gs[1,:])
    a.plot(r.step_s,r.max_stack_error_C,'o-',color=BLUE,lw=1.2,ms=4)
    a.set(xlabel='Thermal step (s)',ylabel='Max. temperature error (°C)',title='a  Fixed-input thermal replay')
    a.set_xticks([.05,.1,.25,.5]);a.set_ylim(0,.07);a.grid(axis='y',alpha=.2)
    z=c[c.thermal_step_s>.05]
    b.plot(z.thermal_step_s,z.max_abs_peak_delta_C,'s-',color=TEAL,lw=1.2,ms=4)
    b.set(xlabel='Thermal step (s)',ylabel='Max. |peak difference| (°C)',title='b  Closed-loop thermal refinement')
    b.set_xticks([.1,.25,.5]);b.set_ylim(0,.27);b.grid(axis='y',alpha=.2)
    s=pd.read_csv(DATA/'thermal_sensitivity_paired.csv')
    order=['Cstack_0p8','Cstack_1p2','Cair_0p5','Cair_0p8','Cair_1p2','Cair_1p5',
           'UA_0p8','UA_1p2','floor_0p10','floor_0p20']
    labels=['Stack C ×0.8','Stack C ×1.2','Second-node C ×0.5',
        'Second-node C ×0.8','Second-node C ×1.2','Second-node C ×1.5',
        'Forced UA ×0.8','Forced UA ×1.2','Min. command 0.10','Min. command 0.20']
    for i,v in enumerate(order):
        d=s[s.variant==v].sort_values('scenario_seed');x=d.delta_peak_temperature_C.to_numpy()
        offsets=np.linspace(-.16,.16,len(x)); color=RED if v=='UA_0p8' else BLUE
        ax.scatter(x,i+offsets,c=color,s=10,alpha=.55,linewidths=0)
        ax.scatter([x.mean()],[i],c=color,marker='D',s=26,zorder=4,edgecolors='white',linewidths=.4)
        for j,row in enumerate(d.itertuples()):
            if row.duration_s<1200:ax.scatter(row.delta_peak_temperature_C,i+offsets[j],marker='^',
                s=36,facecolors='white',edgecolors=RED,zorder=5)
    ax.axvline(0,color='#75818B',ls='--',lw=.7)
    ax.set_yticks(range(len(order)),labels);ax.invert_yaxis()
    ax.set(xlabel='Paired peak-temperature change from nominal parameters (°C)',
           title='c  One-factor sensitivity: 20 safe-PI scenarios per setting',xlim=(-1.8,8.2))
    ax.grid(axis='x',alpha=.2)
    fig.text(.17,.02,'Dots: individual scenarios   Diamonds: means   Open triangle: termination-censored peak',fontsize=7)
    export(fig,'FigureS5_numerical_sensitivity')

def fair():
    d=pd.read_csv(DATA/'holdout_episodes.csv'); chosen=pd.read_csv(DATA/'selected_alarm.csv').threshold_C.iloc[0]
    fine=pd.read_csv(DATA/'refined_holdout_episodes.csv')
    cis=pd.read_csv(DATA/'holdout_paired_bootstrap.csv')
    finer=pd.read_csv(DATA/'refined_holdout_paired_bootstrap.csv')
    selected=f'alarm_{chosen:g}'
    rows=[('fallback',d,'Default fallback\n0.5 s thermal'),('alarm_63',d,'63 °C alarm\n0.5 s thermal'),
          (selected,d,f'{chosen:g} °C alarm\n0.5 s thermal'),(selected,fine,f'{chosen:g} °C alarm\n0.05 s thermal')]
    fig,axes=plt.subplots(1,2,figsize=(183/25.4,98/25.4),sharey=True)
    fig.subplots_adjust(left=.20,right=.98,bottom=.22,top=.89,wspace=.28)
    for ax,metric,title,xlabel in zip(axes,['peak_temperature_C','fan_energy_Wh'],
        ['a  Thermal effect','b  Auxiliary-energy effect'],['Peak-temperature difference (°C)','Fan-energy difference (Wh)']):
        for i,(variant,data,label) in enumerate(rows):
            p=data.groupby(['scenario_seed','variant'])[metric].mean().unstack().sort_index()
            delta=p.calibrated_p90-p[variant];color=TEAL if i==3 else BLUE
            c=(finer.query("thermal_integration=='fine'") if i==3 else cis)
            c=c[(c.comparator==variant)&(c.metric==metric)].iloc[0]
            ax.scatter(delta,i+np.linspace(-.15,.15,20),s=12,color=color,alpha=.5,linewidths=0,zorder=2)
            ax.plot([c.low95,c.high95],[i,i],color=color,lw=2.1,zorder=3)
            ax.scatter(c.mean_difference,i,marker='D',s=32,color=color,edgecolor='white',lw=.5,zorder=4)
        ax.axvline(0,color='#697680',ls='--',lw=.8);ax.set(xlabel=xlabel,title=title)
        ax.grid(axis='x',alpha=.2);ax.set_ylim(3.55,-.55)
    axes[0].set_yticks(range(4),[r[2] for r in rows])
    fig.text(.20,.08,'P90 minus comparator: positive values indicate a higher peak or greater fan energy.',fontsize=7)
    fig.text(.20,.035,'Dots: 20 scenario means   Diamonds: overall means   Bars: paired 95% bootstrap intervals',fontsize=7)
    export(fig,'FigureS6_energy_targeted_alarm')

if __name__=='__main__':
    import sys
    {'thermal':thermal,'fair':fair}[sys.argv[1]]()
