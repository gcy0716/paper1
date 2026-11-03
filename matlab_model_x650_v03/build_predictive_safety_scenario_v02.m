function [p,meta]=build_predictive_safety_scenario_v02(seed,splitName)
% Deterministic, reproducible scenario factory for predictive-safety validation.
% final_test scenarios must not be evaluated until controller settings are locked.
arguments
    seed (1,1) double
    splitName (1,1) string {mustBeMember(splitName,["validation","final_test","ablation"])}
end
rng(seed,'twister'); p=parameters_robust_v05();
scenarioType=mod(seed,4)+1;
switch scenarioType
    case 1 % broad nominal/interpolation
        altitude=500+4500*rand; ambientOffset=-5+10*rand;
        loadScale=.94+.12*rand; uaScale=.92+.16*rand; capScale=.92+.16*rand;
        fcScale=.91+.09*rand; fanScale=.94+.12*rand; taskScale=.94+.12*rand;
    case 2 % high load and power derating
        altitude=3500+2200*rand; ambientOffset=-3+9*rand;
        loadScale=1.05+.10*rand; uaScale=.88+.14*rand; capScale=.88+.14*rand;
        fcScale=.84+.09*rand; fanScale=.95+.15*rand; taskScale=1.05+.15*rand;
    case 3 % hot ambient and cooling degradation
        altitude=1500+4000*rand; ambientOffset=5+7*rand;
        loadScale=.98+.12*rand; uaScale=.78+.14*rand; capScale=.85+.15*rand;
        fcScale=.89+.09*rand; fanScale=1.03+.17*rand; taskScale=1.00+.15*rand;
    otherwise % combined edge case
        altitude=4800+1200*rand; ambientOffset=4+8*rand;
        loadScale=1.08+.10*rand; uaScale=.76+.12*rand; capScale=.78+.14*rand;
        fcScale=.82+.09*rand; fanScale=1.08+.14*rand; taskScale=1.10+.16*rand;
end
p.env.altitudes_m=altitude;
p.mission.vz=p.mission.vz*taskScale.*(.90+.20*rand(size(p.mission.vz)));
p.mission.vh=p.mission.vh*taskScale.*(.90+.20*rand(size(p.mission.vh)));
p.mission.maneuver=min(1.5,p.mission.maneuver*taskScale*(.85+.30*rand));
p.thermal.Tstack0_C=42+6*rand; p.thermal.Tair0_C=27+6*rand; p.bat.SOC0=.72+.16*rand;
dcdcScale=.975+.025*rand;
p.scenario=struct('ambientOffset_C',ambientOffset,'loadScale',loadScale, ...
    'uaScale',uaScale,'capScale',capScale,'fcPowerScale',fcScale, ...
    'dcdcEtaScale',dcdcScale,'fanPowerScale',fanScale);
meta=table(seed,splitName,scenarioType,altitude,ambientOffset,loadScale,uaScale, ...
    capScale,fcScale,dcdcScale,fanScale,taskScale,p.thermal.Tstack0_C, ...
    p.thermal.Tair0_C,p.bat.SOC0, ...
    'VariableNames',{'seed','split','scenario_type','altitude_m','ambient_offset_C', ...
    'load_scale','ua_scale','heat_capacity_scale','fc_power_scale', ...
    'dcdc_efficiency_scale','fan_power_scale','task_scale', ...
    'initial_stack_temperature_C','initial_air_temperature_C','initial_SOC'});
end
