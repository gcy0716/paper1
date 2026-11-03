function evaluate_sac_predictive_safety_v01()
% Frozen-policy comparison: legacy SAC safety versus calibrated P90 + legacy fallback.
root=fileparts(mfilename('fullpath')); p0=parameters_robust_v05();
for seed=[11 22 33]
    agentFile=fullfile('multiseed_v05_finetuned',sprintf('seed_%d',seed), ...
        sprintf('sac_seed_%d_finetuned_5episodes_v05.mat',seed));
    p=p0; p.control.usePredictiveSafety=false;
    evaluate_sac_pretrain_v03(agentFile,sprintf('v05_seed%d_legacy_safety',seed),p,"test_v04");
    p.control.usePredictiveSafety=true; p.control.predictiveMargin_C=2;
    evaluate_sac_pretrain_v03(agentFile,sprintf('v05_seed%d_p90_safety',seed),p,"test_v04");
end

out=fullfile(root,'results','sac_predictive_safety_v01'); if ~exist(out,'dir'), mkdir(out); end
combined=table();
for seed=[11 22 33]
    for variant=["legacy_safety","p90_safety"]
        tag=sprintf('v05_seed%d_%s',seed,variant);
        T=readtable(fullfile(root,'results',['sac_' tag '_test_comparison_v03.csv']),'TextType','string');
        policy="sac_"+tag; T=T(T.policy==policy,:); T.agent_seed=repmat(seed,height(T),1); T.safety_variant=repmat(variant,height(T),1);
        combined=[combined;T]; %#ok<AGROW>
    end
end
writetable(combined,fullfile(out,'sac_predictive_safety_episodes_v01.csv'));
summary=groupsummary(combined,{'agent_seed','safety_variant'},{'mean','std'}, ...
    {'peak_temperature_C','fan_energy_Wh','hydrogen_g','unmet_energy_kJ','final_SOC', ...
    'violation_steps','safety_override_steps','predictive_override_steps','reactive_override_steps'});
writetable(summary,fullfile(out,'sac_predictive_safety_summary_v01.csv'));
disp(summary); fprintf('SAC_PREDICTIVE_SAFETY_V01_COMPLETE\n');
end
