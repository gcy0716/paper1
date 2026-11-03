function results=compare_predictive_safety_v01()
% Compare reactive safe PI with calibrated 10 s P90 warning + reactive fallback.
root=fileparts(mfilename('fullpath')); out=fullfile(root,'results','predictive_safety_v01');
if ~exist(out,'dir'), mkdir(out); end
source=readtable(fullfile(root,'results','sac_v04_ft_seed11_test_comparison_v03.csv'),'TextType','string');
source=source(source.policy=="mpc",:); controllers=["safe_pi","risk_safe_pi"];
results=table(); trajectories=table();
for i=1:height(source)
    baselineReactiveTimes=[];
    for controller=controllers
        p=parameters_robust_v05(); p.env.altitudes_m=source.altitude_m(i);
        standardTamb=atmosphereTemperature(source.altitude_m(i),p);
        p.scenario=struct('ambientOffset_C',source.ambient_C(i)-standardTamb, ...
            'loadScale',source.load_scale(i),'uaScale',source.ua_scale(i), ...
            'capScale',source.heat_capacity_scale(i),'fcPowerScale',source.fc_power_scale(i), ...
            'dcdcEtaScale',source.dcdc_efficiency_scale(i),'fanPowerScale',source.fan_power_scale(i));
        [r,s]=run_x650_v03(controller,p,false);
        r=[table(repmat(source.seed(i),height(r),1),'VariableNames',{'scenario_seed'}),r];
        trajectories=[trajectories;r]; %#ok<AGROW>
        if controller=="safe_pi", baselineReactiveTimes=r.time_s(r.reactive_override); end
        firstPredict=find(r.predictive_override,1,'first');
        if isempty(firstPredict), predictiveLead=nan; else
            nextBaseline=baselineReactiveTimes(baselineReactiveTimes>=r.time_s(firstPredict));
            if isempty(nextBaseline), predictiveLead=nan; else, predictiveLead=nextBaseline(1)-r.time_s(firstPredict); end
        end
        row=table(source.seed(i),controller,s.peak_stack_temperature_C,s.fan_energy_Wh, ...
            s.hydrogen_used_g,s.final_SOC,s.unmet_energy_kJ,s.predictive_override_steps, ...
            s.reactive_override_steps,s.safety_override_steps,predictiveLead, ...
            'VariableNames',{'scenario_seed','controller','peak_temperature_C','fan_energy_Wh', ...
            'hydrogen_g','final_SOC','unmet_energy_kJ','predictive_override_steps', ...
            'reactive_override_steps','total_override_steps','lead_to_next_baseline_reactive_s'});
        results=[results;row]; %#ok<AGROW>
    end
end
writetable(results,fullfile(out,'predictive_safety_comparison_v01.csv'));
writetable(trajectories,fullfile(out,'predictive_safety_trajectories_v01.csv'));

wide=unstack(results,{'peak_temperature_C','fan_energy_Wh','hydrogen_g','final_SOC', ...
    'unmet_energy_kJ','predictive_override_steps','reactive_override_steps','total_override_steps'}, ...
    'controller','GroupingVariables','scenario_seed');
writetable(wide,fullfile(out,'predictive_safety_paired_v01.csv'));
disp(results); fprintf('PREDICTIVE_SAFETY_V01_COMPLETE\n');
end

function Tamb=atmosphereTemperature(h,p)
Tk=p.env.T0_K-p.env.lapse_Kpm*h; Tamb=Tk-273.15;
end
