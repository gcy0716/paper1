function results=evaluate_sac_pretrain_v03(agentFile,tag,p,split)
% Diagnostic evaluation only; short training is not a final agent.
if nargin<1, agentFile='sac_agent_pretrain_v03.mat'; end
if nargin<2, tag='pretrain'; end
if nargin<3 || isempty(p), p=parameters_x650_v03(); end
if nargin<4, split="test"; end
root=fileparts(mfilename('fullpath')); out=fullfile(root,'results');
s=load(fullfile(out,agentFile),'agent'); agent=s.agent;
agent.UseExplorationPolicy=false;
agentPolicy="sac_"+string(tag);
policies=["fixed","pi","mpc",agentPolicy];
seeds=[101 202 303 404 505 606 707 808 909 1010];
results=table();
for seed=seeds
    for policy=policies
        rng(seed,'twister'); env=create_pemfc_rl_env_v03(split,seed,p);
        obs=reset(env); done=false; peakTs=-inf; integral=0; actionVariation=0; previousAction=.4; info=[];
        while ~done
            switch policy
                case "fixed"
                    action=.5;
                case "pi"
                    error=obs(1)*15; integral=min(max(integral+error*.5,-1000),1000);
                    action=min(max(.35+.045*error+.0012*integral,.15),1);
                case "mpc"
                    action=mpcAction(info,previousAction,p);
                otherwise
                    a=getAction(agent,{obs}); action=double(a{1});
            end
            actionVariation=actionVariation+abs(action-previousAction); previousAction=action;
            [obs,reward,done,info]=step(env,action); %#ok<ASGLU>
            peakTs=max(peakTs,info.Ts);
        end
        energyAdequacy=info.servedLoadWh/max(info.totalLoadWh,eps);
        equivalentServedTimeMin=(info.k*p.sim.dt/60)*energyAdequacy;
        unmetEquivalentTimeMin=(info.k*p.sim.dt/60)*(1-energyAdequacy);
        servedEnergyPerMass=info.servedLoadWh/p.mass.takeoff_kg;
        row=table(policy,seed,info.h,info.Tamb,info.loadScale,info.uaScale,info.capScale, ...
            info.fcScale,info.dcdcScale,info.fanPowerScale, ...
            peakTs,info.rewardSum,info.fanWh,info.h2g,info.unmetKJ,info.soc, ...
            info.violations,info.overrides,info.predictiveOverrides,info.reactiveOverrides,actionVariation, ...
            info.demandWh,info.totalLoadWh,info.servedLoadWh,info.fcDcWh, ...
            info.batteryDischargeWh,info.batteryChargeWh,energyAdequacy, ...
            equivalentServedTimeMin,unmetEquivalentTimeMin,servedEnergyPerMass, ...
            info.temperatureCost,info.fanCost,info.smoothCost,info.unmetCost, ...
            'VariableNames',{'policy','seed','altitude_m','ambient_C','load_scale','ua_scale', ...
            'heat_capacity_scale','fc_power_scale','dcdc_efficiency_scale','fan_power_scale', ...
            'peak_temperature_C','total_reward','fan_energy_Wh', ...
            'hydrogen_g','unmet_energy_kJ','final_SOC','violation_steps', ...
            'safety_override_steps','predictive_override_steps','reactive_override_steps','action_total_variation', ...
            'demand_energy_Wh','total_load_energy_Wh','served_load_energy_Wh', ...
            'fc_dc_energy_Wh','battery_discharge_energy_Wh','battery_charge_energy_Wh', ...
            'mission_energy_adequacy','equivalent_fully_served_time_min', ...
            'unmet_equivalent_time_min','served_energy_Wh_per_kg', ...
            'temperature_cost_sum','fan_cost_sum','smooth_cost_sum','unmet_cost_sum'});
        results=[results;row]; %#ok<AGROW>
    end
end
writetable(results,fullfile(out,['sac_' char(tag) '_test_comparison_v03.csv']));
summary=groupsummary(results,'policy',{'mean','std'}, ...
    {'peak_temperature_C','total_reward','fan_energy_Wh','hydrogen_g','unmet_energy_kJ', ...
    'violation_steps','safety_override_steps','predictive_override_steps','reactive_override_steps','action_total_variation', ...
    'mission_energy_adequacy','equivalent_fully_served_time_min','served_energy_Wh_per_kg'});
writetable(summary,fullfile(out,['sac_' char(tag) '_test_summary_v03.csv']));
disp(summary); fprintf('SAC_PRETRAIN_EVALUATION_COMPLETE\n');
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
        cost=cost+((Ts-p.thermal.Tref_C)/p.mpc.temperatureScale_C)^2 ...
            +p.mpc.fanEnergyWeight*candidate^3;
        if Ts>p.thermal.Tmax_C
            cost=cost+p.mpc.constraintPenalty*(Ts-p.thermal.Tmax_C)^2;
        end
    end
    if cost<bestCost, bestCost=cost; best=candidate; end
end
end
