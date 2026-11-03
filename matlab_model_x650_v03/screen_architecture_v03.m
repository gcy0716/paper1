function results=screen_architecture_v03()
% Conceptual architecture screen. Capacity-to-mass and thermal scaling are
% explicit assumptions for sensitivity analysis, not measured hardware data.
root=fileparts(mfilename('fullpath')); out=fullfile(root,'results');
base=parameters_x650_v03();
ratings_W=[1100 1300 1500 1600];
battery_Ah=[5 7.5 10];
controllers=["fixed" "safe_pi" "mpc"];
screenAltitude_m=6000;

fixedMass_kg=base.mass.takeoff_kg-base.mass.fcSystem_kg-base.mass.bufferBattery_kg;
fcSpecificPower_Wpkg=1100/base.mass.fcSystem_kg;       % ASSUMPTION
batterySpecificEnergy_Whpkg=base.bat.Eusable_Wh/base.mass.bufferBattery_kg; % ASSUMPTION
results=table();

for rating=ratings_W
    capacityScale=rating/1100;
    for capacityAh=battery_Ah
        batteryWh=22.2*capacityAh;
        fcMass=rating/fcSpecificPower_Wpkg;
        batteryMass=batteryWh/batterySpecificEnergy_Whpkg;
        takeoffMass=fixedMass_kg+fcMass+batteryMass;
        massFeasible=takeoffMass<=base.x650.takeoffLimit_kg;
        if ~massFeasible
            for controller=controllers
                row=table(rating,capacityAh,batteryWh,fcMass,batteryMass,takeoffMass, ...
                    controller,screenAltitude_m,NaN,NaN,NaN,NaN,NaN,NaN,NaN,false,false,false,false, ...
                    'VariableNames',fieldNames());
                results=[results;row]; %#ok<AGROW>
            end
            continue
        end

        p=base; p.env.altitudes_m=screenAltitude_m;
        p.fc.capacityScale=capacityScale;
        p.fc.PratedNet_W=rating;
        p.fc.ramp_Wps=base.fc.ramp_Wps*capacityScale;
        p.dcdc.Prated_W=2000;
        p.bat.Eusable_Wh=batteryWh;
        p.mass.fcSystem_kg=fcMass; p.mass.bufferBattery_kg=batteryMass;
        p.mass.takeoff_kg=takeoffMass;
        % Modular first-order thermal/cooling scaling assumptions.
        p.thermal.Cstack_JpK=base.thermal.Cstack_JpK*capacityScale;
        p.thermal.Cair_JpK=base.thermal.Cair_JpK*capacityScale;
        p.thermal.UA_stack_air_max_WpK=base.thermal.UA_stack_air_max_WpK*capacityScale^.8;
        p.thermal.UA_air_amb_max_WpK=base.thermal.UA_air_amb_max_WpK*capacityScale^.8;
        p.fan.Prated_W=base.fan.Prated_W*capacityScale;

        for controller=controllers
            [~,s]=run_x650_v03(controller,p,false);
            powerToWeight=(rating/1000)/takeoffMass;
            energyFeasible=s.unmet_energy_kJ<0.1 && s.final_SOC>p.bat.SOCmin+0.01;
            thermalFeasible=s.peak_stack_temperature_C<=p.thermal.Tmax_C;
            missionFeasible=massFeasible && energyFeasible && thermalFeasible;
            row=table(rating,capacityAh,batteryWh,fcMass,batteryMass,takeoffMass, ...
                controller,screenAltitude_m,s.peak_stack_temperature_C,s.fan_energy_Wh, ...
                s.hydrogen_used_g,s.unmet_energy_kJ,s.final_SOC,s.mission_energy_adequacy, ...
                powerToWeight,massFeasible,energyFeasible,thermalFeasible,missionFeasible, ...
                'VariableNames',fieldNames());
            results=[results;row]; %#ok<AGROW>
        end
    end
end

writetable(results,fullfile(out,'architecture_screen_v03.csv'));
assumptions=table(fcSpecificPower_Wpkg,batterySpecificEnergy_Whpkg,base.x650.takeoffLimit_kg, ...
    screenAltitude_m,base.fc.PratedNet_W,base.bat.Eusable_Wh, ...
    'VariableNames',{'assumed_fc_specific_power_W_per_kg','assumed_battery_specific_energy_Wh_per_kg', ...
    'takeoff_mass_limit_kg','screen_altitude_m','baseline_fc_rating_W','baseline_battery_energy_Wh'});
writetable(assumptions,fullfile(out,'architecture_screen_assumptions_v03.csv'));
fprintf('ARCHITECTURE_SCREEN_COMPLETE rows=%d feasible=%d\n',height(results),sum(results.mission_feasible));
end

function names=fieldNames()
names={'fc_rating_W','battery_capacity_Ah','battery_energy_Wh','fc_mass_kg', ...
    'battery_mass_kg','takeoff_mass_kg','controller','altitude_m', ...
    'peak_stack_temperature_C','fan_energy_Wh','hydrogen_used_g','unmet_energy_kJ', ...
    'final_SOC','mission_energy_adequacy','fc_power_to_weight_kW_per_kg', ...
    'mass_feasible','energy_feasible','thermal_feasible','mission_feasible'};
end
