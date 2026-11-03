function results=evaluate_quantile_ablation_v01()
% E1 ablation: legacy reactive protection versus P50/P90 predictive layers.
% Uses dedicated seeds 4101-4120 and never accesses the locked final-test set.
root=fileparts(mfilename('fullpath'));
out=fullfile(root,'results','quantile_ablation_v01');
if ~exist(out,'dir'), mkdir(out); end

agentSeeds=[11 22 33]; agents=cell(size(agentSeeds));
for j=1:numel(agentSeeds)
    seed=agentSeeds(j);
    file=fullfile(root,'results','multiseed_v05_finetuned',sprintf('seed_%d',seed), ...
        sprintf('sac_seed_%d_finetuned_5episodes_v05.mat',seed));
    S=load(file,'agent'); S.agent.UseExplorationPolicy=false; agents{j}=S.agent;
end

scenarioSeeds=4101:4120; variants=["legacy","p50","p90"];
results=table();
manifest=table();
for scenarioSeed=scenarioSeeds
    [p0,meta]=build_predictive_safety_scenario_v02(scenarioSeed,"ablation");
    manifest=[manifest;meta]; %#ok<AGROW>
    for j=1:numel(agentSeeds)
        for variant=variants
            p=p0; p.control.predictiveMargin_C=2;
            p.control.usePredictiveSafety=variant~="legacy";
            if variant=="p50", p.control.predictiveQuantile=.5;
            else, p.control.predictiveQuantile=.9; end
            policy="sac_seed"+string(agentSeeds(j))+"_"+variant;
            row=runEpisode(agents{j},policy,p,scenarioSeed,agentSeeds(j),meta.scenario_type,variant);
            results=[results;row]; %#ok<AGROW>
        end
    end
end

writetable(manifest,fullfile(out,'quantile_ablation_manifest_v01.csv'));
writetable(results,fullfile(out,'quantile_ablation_episodes_v01.csv'));
summary=groupsummary(results,{'variant','agent_seed'},{'mean','std'}, ...
    {'peak_temperature_C','fan_energy_Wh','hydrogen_g','final_SOC','unmet_energy_kJ', ...
    'violation_steps','safety_override_steps','predictive_override_steps','reactive_override_steps'});
writetable(summary,fullfile(out,'quantile_ablation_summary_v01.csv'));
protocol=table("4101-4120",20,3,3,63,65,10000,false, ...
    'VariableNames',{'scenario_seeds','n_scenarios','n_agent_seeds','n_variants', ...
    'trigger_temperature_C','hard_limit_C','planned_bootstrap_resamples','final_test_reused'});
writetable(protocol,fullfile(out,'quantile_ablation_protocol_v01.csv'));
disp(summary); fprintf('QUANTILE_ABLATION_V01_COMPLETE rows=%d\n',height(results));
end

function row=runEpisode(agent,policy,p,scenarioSeed,agentSeed,scenarioType,variant)
env=create_pemfc_rl_env_v03("configured",scenarioSeed,p); obs=reset(env); done=false; info=[];
peak=-inf; previousAction=.4; actionTV=0;
while ~done
    a=getAction(agent,{obs}); action=double(a{1});
    actionTV=actionTV+abs(action-previousAction); previousAction=action;
    [obs,~,done,info]=step(env,action); peak=max(peak,info.Ts);
end
row=table(scenarioSeed,scenarioType,policy,agentSeed,variant,peak,info.fanWh,info.h2g,info.soc, ...
    info.unmetKJ,info.violations,info.overrides,info.predictiveOverrides,info.reactiveOverrides,actionTV, ...
    'VariableNames',{'scenario_seed','scenario_type','policy','agent_seed','variant', ...
    'peak_temperature_C','fan_energy_Wh','hydrogen_g','final_SOC','unmet_energy_kJ', ...
    'violation_steps','safety_override_steps','predictive_override_steps', ...
    'reactive_override_steps','action_total_variation'});
end
