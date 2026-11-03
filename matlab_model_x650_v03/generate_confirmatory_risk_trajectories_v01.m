function allRows=generate_confirmatory_risk_trajectories_v01(nCandidates)
% Reproduce the locked pilot distribution and save trajectories for a
% controller-stratified, near-threshold confirmatory prediction test.
if nargin<1, nCandidates=24; end
root=fileparts(mfilename('fullpath')); out=fullfile(root,'results','confirmatory_prediction_v01');
if ~exist(out,'dir'), mkdir(out); end
rng(20260914,'twister'); controllers=["fixed","safe_pi","mpc"]; allRows=table(); manifest=table();
for candidate=1:nCandidates
    p=parameters_robust_v05(); altitude=4200+1800*rand; ambientOffset=6+10*rand; loadScale=1.02+.18*rand;
    uaScale=.60+.28*rand; capScale=.72+.24*rand; fcScale=.84+.14*rand; fanScale=1.00+.18*rand; taskScale=1.03+.18*rand;
    p.env.altitudes_m=altitude; p.thermal.Tstack0_C=48+9*rand; p.thermal.Tair0_C=30+7*rand; p.bat.SOC0=.76+.10*rand;
    p.mission.vz=p.mission.vz*taskScale; p.mission.vh=p.mission.vh*taskScale; p.mission.maneuver=min(1.5,p.mission.maneuver*taskScale);
    dcdc=.975+.02*rand; p.scenario=struct('ambientOffset_C',ambientOffset,'loadScale',loadScale,'uaScale',uaScale,'capScale',capScale,'fcPowerScale',fcScale,'dcdcEtaScale',dcdc,'fanPowerScale',fanScale);
    manifest=[manifest;table(candidate,altitude,ambientOffset,loadScale,uaScale,capScale,fcScale,dcdc,fanScale,taskScale,p.thermal.Tstack0_C,p.thermal.Tair0_C,p.bat.SOC0, ...
        'VariableNames',{'candidate','altitude_m','ambient_offset_C','load_scale','ua_scale','cap_scale','fc_scale','dcdc_scale','fan_scale','task_scale','initial_stack_temperature_C','initial_air_temperature_C','initial_SOC'})]; %#ok<AGROW>
    for controller=controllers
        [r,~]=run_x650_v03(controller,p,false); n=height(r);
        part=[table(repmat(candidate,n,1),repmat(controller,n,1),'VariableNames',{'candidate','source_controller'}),r];
        allRows=[allRows;part]; %#ok<AGROW>
    end
    fprintf('CONFIRMATORY_TRAJECTORY candidate=%d/%d\n',candidate,nCandidates);
end
writetable(manifest,fullfile(out,'confirmatory_scenario_manifest_v01.csv'));
writetable(allRows,fullfile(out,'confirmatory_trajectories_v01.csv'));
fprintf('CONFIRMATORY_RISK_TRAJECTORIES_V01_COMPLETE rows=%d\n',height(allRows));
end
