function results=screen_predictive_margin_v01()
% Validation-style sensitivity screen for the predictive safety margin.
root=fileparts(mfilename('fullpath')); out=fullfile(root,'results','predictive_safety_v01');
source=readtable(fullfile(root,'results','sac_v04_ft_seed11_test_comparison_v03.csv'),'TextType','string');
source=source(source.policy=="mpc",:); margins=[1 1.5 2 2.5 3]; results=table();
for i=1:height(source)
    p0=parameters_robust_v05(); p0.env.altitudes_m=source.altitude_m(i);
    standardTamb=atmosphereTemperature(source.altitude_m(i),p0);
    p0.scenario=struct('ambientOffset_C',source.ambient_C(i)-standardTamb, ...
        'loadScale',source.load_scale(i),'uaScale',source.ua_scale(i), ...
        'capScale',source.heat_capacity_scale(i),'fcPowerScale',source.fc_power_scale(i), ...
        'dcdcEtaScale',source.dcdc_efficiency_scale(i),'fanPowerScale',source.fan_power_scale(i));
    [~,base]=run_x650_v03("safe_pi",p0,false);
    for margin=margins
        p=p0; p.control.predictiveMargin_C=margin;
        [~,s]=run_x650_v03("risk_safe_pi",p,false);
        row=table(source.seed(i),margin,s.peak_stack_temperature_C,s.fan_energy_Wh, ...
            s.hydrogen_used_g,s.final_SOC,s.unmet_energy_kJ,s.predictive_override_steps, ...
            s.reactive_override_steps,s.peak_stack_temperature_C-base.peak_stack_temperature_C, ...
            s.fan_energy_Wh-base.fan_energy_Wh, ...
            'VariableNames',{'scenario_seed','predictive_margin_C','peak_temperature_C', ...
            'fan_energy_Wh','hydrogen_g','final_SOC','unmet_energy_kJ', ...
            'predictive_override_steps','reactive_override_steps', ...
            'delta_peak_temperature_C','delta_fan_energy_Wh'});
        results=[results;row]; %#ok<AGROW>
    end
end
writetable(results,fullfile(out,'predictive_margin_screen_v01.csv'));
summary=groupsummary(results,'predictive_margin_C',{'mean','min','max'}, ...
    {'delta_peak_temperature_C','delta_fan_energy_Wh','predictive_override_steps','unmet_energy_kJ'});
writetable(summary,fullfile(out,'predictive_margin_screen_summary_v01.csv'));
disp(summary); fprintf('PREDICTIVE_MARGIN_SCREEN_V01_COMPLETE\n');
end

function Tamb=atmosphereTemperature(h,p)
Tk=p.env.T0_K-p.env.lapse_Kpm*h; Tamb=Tk-273.15;
end
