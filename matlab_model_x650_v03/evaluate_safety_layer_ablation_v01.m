function results=evaluate_safety_layer_ablation_v01()
% E3: no protection vs reactive protection vs P90+reactive protection.
root=fileparts(mfilename('fullpath')); out=fullfile(root,'results','safety_layer_ablation_v01');
if ~exist(out,'dir'), mkdir(out); end
agentSeeds=[11 22 33]; agents=cell(size(agentSeeds));
for j=1:numel(agentSeeds)
    file=fullfile(root,'results','multiseed_v05_finetuned',sprintf('seed_%d',agentSeeds(j)), ...
        sprintf('sac_seed_%d_finetuned_5episodes_v05.mat',agentSeeds(j)));
    S=load(file,'agent'); S.agent.UseExplorationPolicy=false; agents{j}=S.agent;
end
scenarioSeeds=4201:4220; variants=["none","reactive","p90_reactive"];
results=table(); manifest=table();
for scenarioSeed=scenarioSeeds
    [p0,meta]=build_predictive_safety_scenario_v02(scenarioSeed,"ablation"); manifest=[manifest;meta]; %#ok<AGROW>
    for j=1:numel(agentSeeds)
        for variant=variants
            p=p0; p.control.predictiveMargin_C=2; p.control.predictiveQuantile=.9;
            p.control.usePredictiveSafety=variant=="p90_reactive";
            p.control.useReactiveSafety=variant~="none";
            row=runEpisode(agents{j},p,scenarioSeed,agentSeeds(j),meta.scenario_type,variant);
            results=[results;row]; %#ok<AGROW>
        end
    end
end
writetable(manifest,fullfile(out,'safety_layer_ablation_manifest_v01.csv'));
writetable(results,fullfile(out,'safety_layer_ablation_episodes_v01.csv'));
summary=groupsummary(results,{'variant','agent_seed'},{'mean','std','sum'}, ...
    {'peak_temperature_C','fan_energy_Wh','unmet_energy_kJ','violation_steps'});
writetable(summary,fullfile(out,'safety_layer_ablation_summary_v01.csv'));
protocol=table("4201-4220",20,3,3,65,70,false, ...
    'VariableNames',{'scenario_seeds','n_scenarios','n_agent_seeds','n_variants','hard_limit_C','termination_temperature_C','final_test_reused'});
writetable(protocol,fullfile(out,'safety_layer_ablation_protocol_v01.csv'));
fprintf('SAFETY_LAYER_ABLATION_V01_COMPLETE rows=%d violations=%d\n',height(results),sum(results.violation_steps));
end

function row=runEpisode(agent,p,scenarioSeed,agentSeed,scenarioType,variant)
env=create_pemfc_rl_env_v03("configured",scenarioSeed,p); obs=reset(env); done=false; info=[]; peak=-inf; previousAction=.4; actionTV=0; steps=0;
while ~done
    a=getAction(agent,{obs}); action=double(a{1}); actionTV=actionTV+abs(action-previousAction); previousAction=action;
    [obs,~,done,info]=step(env,action); peak=max(peak,info.Ts); steps=steps+1;
end
row=table(scenarioSeed,scenarioType,agentSeed,variant,peak,info.fanWh,info.h2g,info.soc,info.unmetKJ, ...
    info.violations,info.overrides,info.predictiveOverrides,info.reactiveOverrides,actionTV,steps,peak>65,peak>70, ...
    'VariableNames',{'scenario_seed','scenario_type','agent_seed','variant','peak_temperature_C','fan_energy_Wh', ...
    'hydrogen_g','final_SOC','unmet_energy_kJ','violation_steps','safety_override_steps', ...
    'predictive_override_steps','reactive_override_steps','action_total_variation','executed_steps','violated_limit','terminated_over_70C'});
end
