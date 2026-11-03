function results=evaluate_reviewer_alarm_ablation_v01(scenarioSeeds)
% Reviewer-requested paired comparison on new scenarios, not the locked test.
% All variants use the same frozen policies and underlying plant.
if nargin<1, scenarioSeeds=4301:4320; end
root=fileparts(mfilename('fullpath'));
out=fullfile(root,'results','reviewer_alarm_ablation_v01');
if ~exist(out,'dir'), mkdir(out); end
agentSeeds=[11 22 33]; variants=["fallback","alarm63","early_fallback", ...
    "raw_p90","calibrated_p90"];
agents=cell(size(agentSeeds));
for j=1:numel(agentSeeds)
    seed=agentSeeds(j);
    file=fullfile(root,'results','multiseed_v05_finetuned',sprintf('seed_%d',seed), ...
        sprintf('sac_seed_%d_finetuned_5episodes_v05.mat',seed));
    S=load(file,'agent'); S.agent.UseExplorationPolicy=false; agents{j}=S.agent;
end
results=table(); manifest=table();
for scenarioSeed=scenarioSeeds
    [p0,meta]=build_predictive_safety_scenario_v02(scenarioSeed,"ablation");
    manifest=[manifest;meta]; %#ok<AGROW>
    for j=1:numel(agentSeeds)
        for variant=variants
            p=p0; p.control.usePredictiveSafety=false;
            p.control.reactiveMargin_C=1; p.control.reactiveMode="rollout";
            p.control.predictiveQuantile=.9; p.control.predictiveMargin_C=2;
            p.control.predictiveApplyCalibration=true;
            switch variant
                case "alarm63"
                    p.control.reactiveMode="alarm";
                    p.control.reactiveMargin_C=2;
                case "early_fallback"
                    p.control.reactiveMargin_C=2;
                case "raw_p90"
                    p.control.usePredictiveSafety=true;
                    p.control.predictiveApplyCalibration=false;
                case "calibrated_p90"
                    p.control.usePredictiveSafety=true;
            end
            row=runEpisode(agents{j},p,scenarioSeed,agentSeeds(j), ...
                meta.scenario_type,variant);
            results=[results;row]; %#ok<AGROW>
        end
    end
    fprintf('REVIEWER_ABLATION_SCENARIO_COMPLETE seed=%d rows=%d\n', ...
        scenarioSeed,height(results));
    writetable(results,fullfile(out,'episodes_partial_v01.csv'));
end
writetable(manifest,fullfile(out,'scenario_manifest_v01.csv'));
writetable(results,fullfile(out,'episodes_v01.csv'));
summary=groupsummary(results,{'variant','agent_seed'},{'mean','std'}, ...
    {'peak_temperature_C','fan_energy_Wh','hydrogen_g','final_SOC', ...
    'unmet_energy_kJ','violation_steps','predictive_override_steps', ...
    'reactive_override_steps'});
writetable(summary,fullfile(out,'summary_v01.csv'));
disp(summary);
end

function row=runEpisode(agent,p,scenarioSeed,agentSeed,scenarioType,variant)
env=create_pemfc_rl_env_v03("configured",scenarioSeed,p);
obs=reset(env); done=false; info=[]; peak=-inf;
while ~done
    a=getAction(agent,{obs});
    [obs,~,done,info]=step(env,double(a{1}));
    peak=max(peak,info.Ts);
end
row=table(scenarioSeed,scenarioType,agentSeed,variant,peak,info.fanWh, ...
    info.h2g,info.soc,info.unmetKJ,info.violations, ...
    info.predictiveOverrides,info.reactiveOverrides, ...
    'VariableNames',{'scenario_seed','scenario_type','agent_seed','variant', ...
    'peak_temperature_C','fan_energy_Wh','hydrogen_g','final_SOC', ...
    'unmet_energy_kJ','violation_steps','predictive_override_steps', ...
    'reactive_override_steps'});
end
