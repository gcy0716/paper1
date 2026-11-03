function results=validate_v05_scenarios()
% Replay the exact ten v04 unseen scenarios for 1.5 and 1.6 kW candidates.
root=fileparts(mfilename('fullpath')); out=fullfile(root,'results');
source=readtable(fullfile(out,'sac_v04_ft_seed11_test_comparison_v03.csv'));
source=source(source.policy=="mpc",:);
controllers=["fixed" "safe_pi" "mpc"];
ratings=[1500 1600]; results=table();
for rating=ratings
    if rating==1500, p0=parameters_recommended_v04(); else, p0=parameters_robust_v05(); end
    for i=1:height(source)
        for controller=controllers
            p=p0; p.env.altitudes_m=source.altitude_m(i);
            standardTamb=atmosphereTemperature(source.altitude_m(i),p);
            p.scenario=struct('ambientOffset_C',source.ambient_C(i)-standardTamb, ...
                'loadScale',source.load_scale(i),'uaScale',source.ua_scale(i), ...
                'capScale',source.heat_capacity_scale(i),'fcPowerScale',source.fc_power_scale(i), ...
                'dcdcEtaScale',source.dcdc_efficiency_scale(i),'fanPowerScale',source.fan_power_scale(i));
            [~,s]=run_x650_v03(controller,p,false);
            feasible=s.unmet_energy_kJ<.1 && s.final_SOC>.26 && s.peak_stack_temperature_C<=65;
            row=table(rating,source.seed(i),controller,s.altitude_m,s.takeoff_mass_kg, ...
                s.peak_stack_temperature_C,s.fan_energy_Wh,s.hydrogen_used_g, ...
                s.unmet_energy_kJ,s.final_SOC,s.mission_energy_adequacy,feasible, ...
                'VariableNames',{'fc_rating_W','scenario_seed','controller','altitude_m', ...
                'takeoff_mass_kg','peak_stack_temperature_C','fan_energy_Wh','hydrogen_used_g', ...
                'unmet_energy_kJ','final_SOC','mission_energy_adequacy','mission_feasible'});
            results=[results;row]; %#ok<AGROW>
        end
    end
end
writetable(results,fullfile(out,'v05_exact_scenario_replay.csv'));
fprintf('V05_REPLAY_COMPLETE feasible_1500=%d feasible_1600=%d\n', ...
    sum(results.fc_rating_W==1500 & results.mission_feasible), ...
    sum(results.fc_rating_W==1600 & results.mission_feasible));
end

function Tamb=atmosphereTemperature(h,p)
Tk=p.env.T0_K-p.env.lapse_Kpm*h; Tamb=Tk-273.15;
end
