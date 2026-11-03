function results=screen_mass_sensitivity_v03()
% Sensitivity of the battery-mass conclusion. Safe PI is used as a rapid,
% conservative architecture screen; final controller ranking is separate.
root=fileparts(mfilename('fullpath')); out=fullfile(root,'results');
base=parameters_x650_v03(); fixedMass=base.mass.takeoff_kg-base.mass.fcSystem_kg-base.mass.bufferBattery_kg;
ratings=[1300 1500]; capacities=[5 7.5];
batterySpecificEnergies=[148 200 250]; massExponents=[1.3 1.5 1.7];
results=table();
for rating=ratings
    scale=rating/1100;
    for ah=capacities
        batteryWh=22.2*ah;
        for specificEnergy=batterySpecificEnergies
            for massExponent=massExponents
                p=base; p.env.altitudes_m=6000; p.fc.capacityScale=scale;
                p.fc.PratedNet_W=rating; p.fc.ramp_Wps=base.fc.ramp_Wps*scale;
                p.dcdc.Prated_W=2000; p.bat.Eusable_Wh=batteryWh;
                p.mass.fcSystem_kg=rating/1100;
                p.mass.bufferBattery_kg=batteryWh/specificEnergy;
                p.mass.takeoff_kg=fixedMass+p.mass.fcSystem_kg+p.mass.bufferBattery_kg;
                p.power.massExponent=massExponent;
                p.thermal.Cstack_JpK=base.thermal.Cstack_JpK*scale;
                p.thermal.Cair_JpK=base.thermal.Cair_JpK*scale;
                p.thermal.UA_stack_air_max_WpK=base.thermal.UA_stack_air_max_WpK*scale^.8;
                p.thermal.UA_air_amb_max_WpK=base.thermal.UA_air_amb_max_WpK*scale^.8;
                p.fan.Prated_W=base.fan.Prated_W*scale;
                massOK=p.mass.takeoff_kg<=p.x650.takeoffLimit_kg;
                if massOK
                    [~,s]=run_x650_v03("safe_pi",p,false);
                    feasible=s.unmet_energy_kJ<.1 && s.final_SOC>.26 && s.peak_stack_temperature_C<=65;
                    unmet=s.unmet_energy_kJ; finalSOC=s.final_SOC; peakT=s.peak_stack_temperature_C;
                else
                    feasible=false; unmet=NaN; finalSOC=NaN; peakT=NaN;
                end
                row=table(rating,ah,specificEnergy,massExponent,p.mass.takeoff_kg,unmet,finalSOC,peakT,massOK,feasible, ...
                    'VariableNames',{'fc_rating_W','battery_capacity_Ah','battery_specific_energy_Wh_per_kg', ...
                    'propulsion_mass_exponent','takeoff_mass_kg','unmet_energy_kJ','final_SOC', ...
                    'peak_stack_temperature_C','mass_feasible','mission_feasible'});
                results=[results;row]; %#ok<AGROW>
            end
        end
    end
end
writetable(results,fullfile(out,'mass_capacity_sensitivity_v03.csv'));
fprintf('MASS_SENSITIVITY_COMPLETE cases=%d feasible=%d\n',height(results),sum(results.mission_feasible));
end
