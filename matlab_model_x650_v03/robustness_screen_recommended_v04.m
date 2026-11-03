function results=robustness_screen_recommended_v04()
% Matched unseen high-altitude scenarios for the recommended architecture.
root=fileparts(mfilename('fullpath')); out=fullfile(root,'results');
p0=parameters_recommended_v04();
controllers=["fixed" "safe_pi" "mpc"];
seeds=[101 202 303 404 505 606 707 808 909 1010 1111 1212];
results=table();
for seed=seeds
    rng(seed,'twister');
    altitude=5000+1000*rand;
    ambientOffset=-5+10*rand;
    loadScale=.95+.15*rand;
    uaScale=.80+.25*rand;
    capScale=.85+.25*rand;
    fcPowerScale=.88+.12*rand;
    dcdcEtaScale=.97+.03*rand;
    fanPowerScale=.90+.20*rand;
    for controller=controllers
        p=p0; p.env.altitudes_m=altitude;
        p.scenario=struct('ambientOffset_C',ambientOffset,'loadScale',loadScale, ...
            'uaScale',uaScale,'capScale',capScale,'fcPowerScale',fcPowerScale, ...
            'dcdcEtaScale',dcdcEtaScale,'fanPowerScale',fanPowerScale);
        [~,s]=run_x650_v03(controller,p,false);
        feasible=s.unmet_energy_kJ<.1 && s.final_SOC>.26 && s.peak_stack_temperature_C<=65;
        row=table(controller,seed,altitude,ambientOffset,loadScale,uaScale,capScale, ...
            fcPowerScale,dcdcEtaScale,fanPowerScale,s.takeoff_mass_kg,s.peak_stack_temperature_C, ...
            s.fan_energy_Wh,s.hydrogen_used_g,s.unmet_energy_kJ,s.final_SOC, ...
            s.mission_energy_adequacy,s.safety_override_steps,feasible, ...
            'VariableNames',{'controller','scenario_seed','altitude_m','ambient_offset_C', ...
            'load_scale','ua_scale','heat_capacity_scale','fc_power_scale','dcdc_efficiency_scale', ...
            'fan_power_scale','takeoff_mass_kg','peak_stack_temperature_C','fan_energy_Wh', ...
            'hydrogen_used_g','unmet_energy_kJ','final_SOC','mission_energy_adequacy', ...
            'safety_override_steps','mission_feasible'});
        results=[results;row]; %#ok<AGROW>
    end
end
writetable(results,fullfile(out,'recommended_architecture_robustness_v04.csv'));
fprintf('RECOMMENDED_ROBUSTNESS_COMPLETE rows=%d feasible=%d\n',height(results),sum(results.mission_feasible));
end
