function [results,summary]=validate_predictive_margin_v02()
% Validation-only margin screen. Does not run final_test scenarios.
root=fileparts(mfilename('fullpath')); out=fullfile(root,'results','predictive_safety_v02');
if ~exist(out,'dir'), mkdir(out); end
validationSeeds=2101:2120; finalTestSeeds=3101:3120; margins=[1 1.5 2 2.5 3];
manifest=table();
for seed=[validationSeeds finalTestSeeds]
    if seed<3000, split="validation"; else, split="final_test"; end
    [~,meta]=build_predictive_safety_scenario_v02(seed,split); manifest=[manifest;meta]; %#ok<AGROW>
end
writetable(manifest,fullfile(out,'scenario_manifest_v02.csv'));

results=table();
for seed=validationSeeds
    [p0,~]=build_predictive_safety_scenario_v02(seed,"validation");
    [~,base]=run_x650_v03("safe_pi",p0,false);
    for margin=margins
        p=p0; p.control.predictiveMargin_C=margin;
        [~,s]=run_x650_v03("risk_safe_pi",p,false);
        row=table(seed,getType(seed),margin,s.peak_stack_temperature_C,s.fan_energy_Wh, ...
            s.hydrogen_used_g,s.final_SOC,s.unmet_energy_kJ,s.predictive_override_steps, ...
            s.reactive_override_steps,s.peak_stack_temperature_C-base.peak_stack_temperature_C, ...
            s.fan_energy_Wh-base.fan_energy_Wh,base.peak_stack_temperature_C,base.fan_energy_Wh, ...
            'VariableNames',{'seed','scenario_type','predictive_margin_C','peak_temperature_C', ...
            'fan_energy_Wh','hydrogen_g','final_SOC','unmet_energy_kJ', ...
            'predictive_override_steps','reactive_override_steps','delta_peak_temperature_C', ...
            'delta_fan_energy_Wh','baseline_peak_temperature_C','baseline_fan_energy_Wh'});
        results=[results;row]; %#ok<AGROW>
    end
end
writetable(results,fullfile(out,'validation_margin_results_v02.csv'));
summary=groupsummary(results,'predictive_margin_C',{'mean','min','max'}, ...
    {'delta_peak_temperature_C','delta_fan_energy_Wh','predictive_override_steps', ...
    'peak_temperature_C','unmet_energy_kJ'});
writetable(summary,fullfile(out,'validation_margin_summary_v02.csv'));

protocol=table(2,65,63,20,20,false, ...
    'VariableNames',{'locked_margin_C','hard_temperature_limit_C','predictive_trigger_C', ...
    'validation_scenarios','reserved_final_test_scenarios','final_test_evaluated'});
writetable(protocol,fullfile(out,'locked_protocol_v02.csv'));
disp(summary); fprintf('PREDICTIVE_MARGIN_VALIDATION_V02_COMPLETE final_test_evaluated=0\n');
end

function value=getType(seed), value=mod(seed,4)+1; end
