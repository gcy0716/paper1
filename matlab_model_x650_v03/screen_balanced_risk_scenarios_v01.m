function results=screen_balanced_risk_scenarios_v01(nCandidates)
% Pilot-only search for near-risk trajectories across fixed, safe-PI and MPC.
% Does not train or select a predictor and does not access final-test seeds.
if nargin<1, nCandidates=24; end
root=fileparts(mfilename('fullpath')); out=fullfile(root,'results','confirmatory_prediction_v01');
if ~exist(out,'dir'), mkdir(out); end
rng(20260914,'twister'); controllers=["fixed","safe_pi","mpc"]; results=table();
for candidate=1:nCandidates
    p=parameters_robust_v05();
    altitude=4200+1800*rand; ambientOffset=6+10*rand; loadScale=1.02+.18*rand;
    uaScale=.60+.28*rand; capScale=.72+.24*rand; fcScale=.84+.14*rand;
    fanScale=1.00+.18*rand; taskScale=1.03+.18*rand;
    p.env.altitudes_m=altitude; p.thermal.Tstack0_C=48+9*rand; p.thermal.Tair0_C=30+7*rand; p.bat.SOC0=.76+.10*rand;
    p.mission.vz=p.mission.vz*taskScale; p.mission.vh=p.mission.vh*taskScale; p.mission.maneuver=min(1.5,p.mission.maneuver*taskScale);
    p.scenario=struct('ambientOffset_C',ambientOffset,'loadScale',loadScale,'uaScale',uaScale, ...
        'capScale',capScale,'fcPowerScale',fcScale,'dcdcEtaScale',.975+.02*rand,'fanPowerScale',fanScale);
    for controller=controllers
        [r,s]=run_x650_v03(controller,p,false); over63=sum(r.stack_temperature_C>63); near=sum(r.stack_temperature_C>=61 & r.stack_temperature_C<=65);
        row=table(candidate,controller,altitude,ambientOffset,loadScale,uaScale,capScale,fcScale,fanScale,taskScale, ...
            p.thermal.Tstack0_C,max(r.stack_temperature_C),over63,near,s.fan_energy_Wh,s.unmet_energy_kJ, ...
            'VariableNames',{'candidate','controller','altitude_m','ambient_offset_C','load_scale','ua_scale','cap_scale','fc_scale','fan_scale','task_scale','initial_temperature_C','peak_temperature_C','points_over_63C','points_61_to_65C','fan_energy_Wh','unmet_energy_kJ'});
        results=[results;row]; %#ok<AGROW>
    end
    fprintf('RISK_SCREEN candidate=%d/%d\n',candidate,nCandidates);
end
writetable(results,fullfile(out,'balanced_risk_candidate_screen_v01.csv'));
summary=groupsummary(results,'controller',{'min','mean','max'},{'peak_temperature_C','points_over_63C','points_61_to_65C','unmet_energy_kJ'});
writetable(summary,fullfile(out,'balanced_risk_candidate_summary_v01.csv')); disp(summary);
fprintf('BALANCED_RISK_SCREEN_V01_COMPLETE rows=%d\n',height(results));
end
