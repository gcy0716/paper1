function results=resolve_worst_case_v05()
% Find the minimum FC rating that clears the remaining exact worst case.
root=fileparts(mfilename('fullpath')); out=fullfile(root,'results');
source=readtable(fullfile(out,'sac_v04_ft_seed11_test_comparison_v03.csv'));
source=source(source.policy=="mpc" & source.seed==505,:);
ratings=[1600 1700 1800]; controllers=["fixed" "safe_pi" "mpc"];
results=table();
for rating=ratings
    p0=scaledParameters(rating);
    for controller=controllers
        p=p0; p.env.altitudes_m=source.altitude_m;
        standardTamb=p.env.T0_K-p.env.lapse_Kpm*source.altitude_m-273.15;
        p.scenario=struct('ambientOffset_C',source.ambient_C-standardTamb, ...
            'loadScale',source.load_scale,'uaScale',source.ua_scale, ...
            'capScale',source.heat_capacity_scale,'fcPowerScale',source.fc_power_scale, ...
            'dcdcEtaScale',source.dcdc_efficiency_scale,'fanPowerScale',source.fan_power_scale);
        [~,s]=run_x650_v03(controller,p,false);
        feasible=s.unmet_energy_kJ<.1 && s.final_SOC>.26 && s.peak_stack_temperature_C<=65;
        row=table(rating,controller,s.takeoff_mass_kg,s.peak_stack_temperature_C, ...
            s.fan_energy_Wh,s.unmet_energy_kJ,s.final_SOC,feasible, ...
            'VariableNames',{'fc_rating_W','controller','takeoff_mass_kg', ...
            'peak_stack_temperature_C','fan_energy_Wh','unmet_energy_kJ','final_SOC','mission_feasible'});
        results=[results;row]; %#ok<AGROW>
    end
end
writetable(results,fullfile(out,'v05_worst_case_power_margin.csv'));
fprintf('V05_WORST_CASE_COMPLETE\n');
end

function p=scaledParameters(rating)
p=parameters_x650_v03(); scale=rating/1100; fixedMass=p.mass.takeoff_kg-1-.75;
p.fc.capacityScale=scale; p.fc.PratedNet_W=rating; p.fc.ramp_Wps=p.fc.ramp_Wps*scale;
p.dcdc.Prated_W=2000; p.mass.fcSystem_kg=rating/1100; p.mass.bufferBattery_kg=.75;
p.mass.takeoff_kg=fixedMass+p.mass.fcSystem_kg+p.mass.bufferBattery_kg;
p.thermal.Cstack_JpK=p.thermal.Cstack_JpK*scale; p.thermal.Cair_JpK=p.thermal.Cair_JpK*scale;
p.thermal.UA_stack_air_max_WpK=p.thermal.UA_stack_air_max_WpK*scale^.8;
p.thermal.UA_air_amb_max_WpK=p.thermal.UA_air_amb_max_WpK*scale^.8;
p.fan.Prated_W=p.fan.Prated_W*scale;
end
