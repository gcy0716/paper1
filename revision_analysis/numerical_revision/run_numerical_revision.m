function run_numerical_revision(stage,agentFilter)
here=fileparts(mfilename('fullpath')); root=fileparts(fileparts(here));
model=fullfile(root,'matlab_model_x650_v03'); addpath(model);
out=fullfile(here,'results'); if ~exist(out,'dir'), mkdir(out); end
if nargin<1, stage="thermal"; end
if nargin<2, agentFilter=[]; end
if stage=="thermal"
    variants=["nominal","Cstack_0p8","Cstack_1p2","Cair_0p5","Cair_0p8", ...
        "Cair_1p2","Cair_1p5","UA_0p8","UA_1p2","floor_0p10","floor_0p20"];
    rows=table(); manifest=table();
    for seed=4301:4320
        [p0,meta]=build_predictive_safety_scenario_v02(seed,"ablation");
        manifest=[manifest;meta]; %#ok<AGROW>
        for variant=variants
            p=p0; p.control.usePredictiveSafety=false;
            switch variant
                case "Cstack_0p8", p.thermal.Cstack_JpK=p.thermal.Cstack_JpK*.8;
                case "Cstack_1p2", p.thermal.Cstack_JpK=p.thermal.Cstack_JpK*1.2;
                case "Cair_0p5", p.thermal.Cair_JpK=p.thermal.Cair_JpK*.5;
                case "Cair_0p8", p.thermal.Cair_JpK=p.thermal.Cair_JpK*.8;
                case "Cair_1p2", p.thermal.Cair_JpK=p.thermal.Cair_JpK*1.2;
                case "Cair_1p5", p.thermal.Cair_JpK=p.thermal.Cair_JpK*1.5;
                case "UA_0p8"
                    p.thermal.UA_stack_air_max_WpK=p.thermal.UA_stack_air_max_WpK*.8;
                    p.thermal.UA_air_amb_max_WpK=p.thermal.UA_air_amb_max_WpK*.8;
                case "UA_1p2"
                    p.thermal.UA_stack_air_max_WpK=p.thermal.UA_stack_air_max_WpK*1.2;
                    p.thermal.UA_air_amb_max_WpK=p.thermal.UA_air_amb_max_WpK*1.2;
                case "floor_0p10", p.fan.minCommand=.10;
                case "floor_0p20", p.fan.minCommand=.20;
            end
            [row,trace]=episode([],p,seed,0,variant,variant=="nominal");
            rows=[rows;row]; %#ok<AGROW>
            if variant=="nominal"
                writetable(trace,fullfile(out,sprintf('thermal_trace_%d.csv',seed)));
                save(fullfile(out,sprintf('thermal_parameters_%d.mat',seed)),'p');
            end
        end
        writetable(rows,fullfile(out,'thermal_sensitivity.csv'));
        fprintf('THERMAL_DONE seed=%d rows=%d\n',seed,height(rows));
    end
    writetable(manifest,fullfile(out,'thermal_manifest.csv'));
elseif stage=="failure_diagnostics"
    rows=table();
    for seed=[4302 4306 4314]
        [p,~]=build_predictive_safety_scenario_v02(seed,"ablation");
        p.control.usePredictiveSafety=false;
        p.thermal.UA_stack_air_max_WpK=p.thermal.UA_stack_air_max_WpK*.8;
        p.thermal.UA_air_amb_max_WpK=p.thermal.UA_air_amb_max_WpK*.8;
        [row,trace]=episode([],p,seed,0,"UA_0p8",true);
        rows=[rows;row]; %#ok<AGROW>
        writetable(trace,fullfile(out,sprintf('failure_trace_%d.csv',seed)));
    end
    writetable(rows,fullfile(out,'failure_diagnostics.csv'));
elseif stage=="refinement"
    rows=table();
    for seed=4301:4320
        [p0,~]=build_predictive_safety_scenario_v02(seed,"ablation");
        for nsub=[1 2 5 10]
            p=p0; p.control.usePredictiveSafety=false; p.sim.thermalSubsteps=nsub;
            [row,~]=episode([],p,seed,0,"thermal_substeps_"+nsub,false);
            row.thermal_step_s=.5/nsub; rows=[rows;row]; %#ok<AGROW>
        end
        writetable(rows,fullfile(out,'closed_loop_thermal_refinement.csv'));
        fprintf('REFINEMENT_DONE seed=%d rows=%d\n',seed,height(rows));
    end
elseif stage=="rating"
    source=readtable(fullfile(model,'results','sac_v04_ft_seed11_test_comparison_v03.csv'));
    source=source(string(source.policy)=="mpc" & source.seed==505,:);
    assert(height(source)==1); rows=table();
    for rating=[1600 1700 1800]
        p=parameters_x650_v03(); scale=rating/1100;
        p.fc.capacityScale=scale; p.fc.PratedNet_W=rating;
        p.fc.ramp_Wps=p.fc.ramp_Wps*scale; p.dcdc.Prated_W=2000;
        p.mass.takeoff_kg=p.mass.takeoff_kg-1+scale; p.mass.fcSystem_kg=scale;
        p.thermal.Cstack_JpK=p.thermal.Cstack_JpK*scale;
        p.thermal.Cair_JpK=p.thermal.Cair_JpK*scale;
        p.thermal.UA_stack_air_max_WpK=p.thermal.UA_stack_air_max_WpK*scale^.8;
        p.thermal.UA_air_amb_max_WpK=p.thermal.UA_air_amb_max_WpK*scale^.8;
        p.fan.Prated_W=p.fan.Prated_W*scale;
        p.env.altitudes_m=source.altitude_m;
        p.scenario=struct('ambientOffset_C',source.ambient_C-(p.env.T0_K-p.env.lapse_Kpm*source.altitude_m-273.15), ...
            'loadScale',source.load_scale,'uaScale',source.ua_scale, ...
            'capScale',source.heat_capacity_scale,'fcPowerScale',source.fc_power_scale, ...
            'dcdcEtaScale',source.dcdc_efficiency_scale,'fanPowerScale',source.fan_power_scale);
        for controller=["fixed","safe_pi"]
            if controller=="fixed", policy=.5; p.control.useReactiveSafety=false;
            else, policy=[]; p.control.useReactiveSafety=true; end
            [row,~]=episode(policy,p,505,0,controller,false);
            row.rating_W=rating; row.takeoff_mass_kg=p.mass.takeoff_kg;
            rows=[rows;row]; %#ok<AGROW>
        end
    end
    writetable(rows,fullfile(out,'rating_current_plant.csv')); disp(rows);
elseif stage=="manifests"
    for group=["development","holdout"]
        if group=="development", ids=4401:4410; else, ids=4501:4520; end
        manifest=table();
        for seed=ids
            [~,meta]=build_predictive_safety_scenario_v02(seed,"ablation");
            manifest=[manifest;meta]; %#ok<AGROW>
        end
        writetable(manifest,fullfile(out,group+"_manifest.csv"));
    end
elseif stage=="benchmark"
    agent=loadAgent(model,11);
    [p,~]=build_predictive_safety_scenario_v02(4301,"ablation");
    p.control.usePredictiveSafety=false; p.control.reactiveMode="alarm";
    p.control.reactiveMargin_C=2;
    tic; [row,~]=episode(agent,p,4301,11,"alarm63",false); seconds=toc;
    writetable(row,fullfile(out,'benchmark_reproduction.csv'));
    fprintf('BENCHMARK_SECONDS %.3f\n',seconds); disp(row);
elseif stage=="development" || stage=="holdout" || stage=="refined_holdout"
    seeds=[11 22 33]; suffix="";
    if ~isempty(agentFilter), seeds=agentFilter; suffix="_agent"+agentFilter; end
    agents=cell(size(seeds)); for j=1:numel(seeds), agents{j}=loadAgent(model,seeds(j)); end
    rows=table();
    if stage=="development", scenarios=4401:4410; thresholds=61:.5:64.5;
    else
        scenarios=4501:4520;
        chosen=readtable(fullfile(out,'selected_alarm.csv'));
        if stage=="refined_holdout", thresholds=chosen.threshold_C(1);
        else, thresholds=unique([chosen.threshold_C(1),63]); end
    end
    for scenario=scenarios
        [p0,~]=build_predictive_safety_scenario_v02(scenario,"ablation");
        for j=1:numel(seeds)
            thresholdList=[NaN,thresholds,Inf];
            if stage=="refined_holdout", thresholdList=[NaN,thresholds]; end
            for threshold=thresholdList
                p=p0; p.control.usePredictiveSafety=false;
                if stage=="refined_holdout", p.sim.thermalSubsteps=10; end
                p.control.reactiveMode="rollout"; p.control.reactiveMargin_C=1;
                if isnan(threshold)
                    p.control.usePredictiveSafety=true;
                    p.control.predictiveApplyCalibration=true;
                    p.control.predictiveQuantile=.9; p.control.predictiveMargin_C=2;
                    variant="calibrated_p90";
                elseif isinf(threshold), variant="fallback";
                else
                    p.control.reactiveMode="alarm";
                    p.control.reactiveMargin_C=65-threshold;
                    variant="alarm_"+string(threshold);
                end
                [row,~]=episode(agents{j},p,scenario,seeds(j),variant,false);
                row.threshold_C=threshold; rows=[rows;row]; %#ok<AGROW>
            end
        end
        writetable(rows,fullfile(out,stage+"_episodes"+suffix+".csv"));
        fprintf('%s_DONE seed=%d rows=%d\n',upper(stage),scenario,height(rows));
    end
else, error('Unknown stage'); end
end

function agent=loadAgent(root,seed)
S=load(fullfile(root,'results','multiseed_v05_finetuned',sprintf('seed_%d',seed), ...
    sprintf('sac_seed_%d_finetuned_5episodes_v05.mat',seed)),'agent');
agent=S.agent; agent.UseExplorationPolicy=false;
end

function [row,trace]=episode(agent,p,scenario,agentSeed,variant,keepTrace)
if isfield(p.sim,'thermalSubsteps')
    env=create_pemfc_thermal_refined("configured",scenario,p);
else
    env=create_pemfc_rl_env_v03("configured",scenario,p);
end
obs=reset(env); done=false; integ=0; peak=p.thermal.Tstack0_C;
data=zeros(round(p.sim.tEnd/p.sim.dt),10); n=0;
while ~done
    if isempty(agent)
        err=obs(1)*15; integ=min(max(integ+err*p.sim.dt,-1000),1000);
        action=min(max(.35+.045*err+.0012*integ,p.fan.minCommand),1);
    elseif isnumeric(agent)
        action=agent;
    else
        a=getAction(agent,{obs}); action=double(a{1});
    end
    [obs,~,done,info]=step(env,action); n=n+1; peak=max(peak,info.Ts);
    if keepTrace
        data(n,:)=[info.k*p.sim.dt,info.Ts,info.Ta,info.qgen,info.fan, ...
            info.Tamb,info.rho,info.uaScale,info.capScale,info.fanWh];
    end
end
row=table(scenario,agentSeed,variant,peak,info.fanWh,info.h2g,info.soc, ...
    info.unmetKJ,info.violations,info.predictiveOverrides,info.reactiveOverrides,n*p.sim.dt, ...
    'VariableNames',{'scenario_seed','agent_seed','variant','peak_temperature_C', ...
    'fan_energy_Wh','hydrogen_g','final_SOC','unmet_energy_kJ','violation_steps', ...
    'predictive_override_steps','reactive_override_steps','duration_s'});
trace=array2table(data(1:n,:),'VariableNames',{'time_s','Ts_C','Ta_C','Qgen_W', ...
    'fan','Tamb_C','rho','uaScale','capScale','fan_Wh'});
end
