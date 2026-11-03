function results=evaluate_unified_validation_v02(mode)
% Unified deterministic scenario evaluation. Default remains validation-only.
if nargin<1, mode="validation"; end
mode=string(mode); assert(any(mode==["validation","final_test"]),'Unknown mode');
root=fileparts(mfilename('fullpath')); out=fullfile(root,'results','predictive_safety_v02');
agentSeeds=[11 22 33]; agents=cell(size(agentSeeds));
for j=1:numel(agentSeeds)
    seed=agentSeeds(j); file=fullfile(root,'results','multiseed_v05_finetuned',sprintf('seed_%d',seed), ...
        sprintf('sac_seed_%d_finetuned_5episodes_v05.mat',seed));
    S=load(file,'agent'); S.agent.UseExplorationPolicy=false; agents{j}=S.agent;
end

if mode=="validation", scenarioSeeds=2101:2120; prefix="unified_validation"; evaluated=0;
else, scenarioSeeds=3101:3120; prefix="unified_final_test"; evaluated=1; end
results=table();
for scenarioSeed=scenarioSeeds
    [p0,meta]=build_predictive_safety_scenario_v02(scenarioSeed,mode);
    % Model-driven baselines, both retaining the legacy safety filter.
    for policy=["safe_pi","mpc"]
        p=p0; p.control.usePredictiveSafety=false;
        row=runEpisode([],policy,p,scenarioSeed,0,meta.scenario_type);
        results=[results;row]; %#ok<AGROW>
    end
    % Frozen SAC policies with legacy-only and P90+legacy safety.
    for j=1:numel(agentSeeds)
        for variant=["legacy","p90"]
            p=p0; p.control.usePredictiveSafety=variant=="p90"; p.control.predictiveMargin_C=2;
            policy="sac_seed"+string(agentSeeds(j))+"_"+variant;
            row=runEpisode(agents{j},policy,p,scenarioSeed,agentSeeds(j),meta.scenario_type);
            results=[results;row]; %#ok<AGROW>
        end
    end
end
writetable(results,fullfile(out,char(prefix+"_episodes_v02.csv")));
summary=groupsummary(results,'policy',{'mean','std'}, ...
    {'peak_temperature_C','fan_energy_Wh','hydrogen_g','final_SOC','unmet_energy_kJ', ...
    'violation_steps','safety_override_steps','predictive_override_steps','reactive_override_steps'});
writetable(summary,fullfile(out,char(prefix+"_summary_v02.csv")));
feasible=results(results.unmet_energy_kJ<.1,:);
feasibleSummary=groupsummary(feasible,'policy',{'mean','std'}, ...
    {'peak_temperature_C','fan_energy_Wh','hydrogen_g','final_SOC', ...
    'violation_steps','safety_override_steps','predictive_override_steps','reactive_override_steps'});
writetable(feasibleSummary,fullfile(out,char(prefix+"_feasible_summary_v02.csv")));
disp(summary); fprintf('UNIFIED_%s_V02_COMPLETE final_test_evaluated=%d\n',upper(char(mode)),evaluated);
end

function row=runEpisode(agent,policy,p,scenarioSeed,agentSeed,scenarioType)
env=create_pemfc_rl_env_v03("configured",scenarioSeed,p); obs=reset(env); done=false; info=[];
peak=-inf; integral=0; previousAction=.4; actionTV=0;
while ~done
    if policy=="safe_pi"
        error=obs(1)*15; integral=min(max(integral+error*p.sim.dt,-1000),1000);
        action=min(max(.35+.045*error+.0012*integral,.15),1);
    elseif policy=="mpc"
        action=mpcAction(info,previousAction,p);
    else
        a=getAction(agent,{obs}); action=double(a{1});
    end
    actionTV=actionTV+abs(action-previousAction); previousAction=action;
    [obs,~,done,info]=step(env,action); peak=max(peak,info.Ts);
end
row=table(scenarioSeed,scenarioType,policy,agentSeed,peak,info.fanWh,info.h2g,info.soc, ...
    info.unmetKJ,info.violations,info.overrides,info.predictiveOverrides,info.reactiveOverrides,actionTV, ...
    'VariableNames',{'scenario_seed','scenario_type','policy','agent_seed','peak_temperature_C', ...
    'fan_energy_Wh','hydrogen_g','final_SOC','unmet_energy_kJ','violation_steps', ...
    'safety_override_steps','predictive_override_steps','reactive_override_steps','action_total_variation'});
end

function best=mpcAction(info,uPrev,p)
if isempty(info), best=.5; return; end
best=1; bestCost=inf; density=(info.rho/p.env.rho0)^.5;
for candidate=p.mpc.commandGrid
    Ts=info.Ts; Ta=info.Ta; cost=p.mpc.moveWeight*(candidate-uPrev)^2;
    for j=1:p.mpc.horizonSteps
        Qsa=p.thermal.UA_stack_air_max_WpK*info.uaScale*candidate^.7*density*(Ts-Ta);
        Qaa=p.thermal.UA_air_amb_max_WpK*info.uaScale*candidate^.7*density*(Ta-info.Tamb);
        Qpass=p.thermal.UA_passive_WpK*(Ts-info.Tamb);
        Ts=Ts+p.sim.dt*(info.qgen-Qsa-Qpass)/(p.thermal.Cstack_JpK*info.capScale);
        Ta=Ta+p.sim.dt*(Qsa-Qaa)/p.thermal.Cair_JpK;
        cost=cost+((Ts-p.thermal.Tref_C)/p.mpc.temperatureScale_C)^2+p.mpc.fanEnergyWeight*candidate^3;
        if Ts>p.thermal.Tmax_C, cost=cost+p.mpc.constraintPenalty*(Ts-p.thermal.Tmax_C)^2; end
    end
    if cost<bestCost, bestCost=cost; best=candidate; end
end
end
