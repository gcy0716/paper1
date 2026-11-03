function manifest = generate_probability_dataset_v01(nTrain,nVal,nTest,nOod,datasetSeed,outputFolder)
% Generate scenario-level isolated trajectories for probabilistic thermal prediction.
% Splits are assigned before simulation; rows from one scenario never cross splits.
if nargin<1, nTrain=48; end
if nargin<2, nVal=12; end
if nargin<3, nTest=12; end
if nargin<4, nOod=12; end

root=fileparts(mfilename('fullpath'));
if nargin<5, datasetSeed=20260901; end
if nargin<6, outputFolder=fullfile(root,'results','probability_dataset_v01'); end
out=outputFolder;
if ~exist(out,'dir'), mkdir(out); end
rng(datasetSeed,'twister');

splitNames=[repmat("train",nTrain,1);repmat("validation",nVal,1); ...
    repmat("test",nTest,1);repmat("ood_test",nOod,1)];
nScenarios=numel(splitNames);
controllers=["fixed","safe_pi","mpc"];
allRows=table(); manifest=table();

for scenarioId=1:nScenarios
    split=splitNames(scenarioId);
    isOod=split=="ood_test";
    p=parameters_robust_v05();
    p.env.altitudes_m=drawRange(isOod,[0 5000],[5000 6000]);

    if isOod
        ambientOffset=drawSigned([7 12]);
        loadScale=drawRange(true,[0 0],[1.08 1.18]);
        uaScale=drawRange(true,[0 0],[0.78 0.90]);
        capScale=drawRange(true,[0 0],[0.78 0.90]);
        fcPowerScale=drawRange(true,[0 0],[0.82 0.91]);
        dcdcScale=drawRange(true,[0 0],[0.965 0.980]);
        fanPowerScale=drawRange(true,[0 0],[1.10 1.22]);
        taskScale=drawRange(true,[0 0],[1.10 1.25]);
    else
        ambientOffset=-5+10*rand;
        loadScale=.92+.16*rand;
        uaScale=.90+.20*rand;
        capScale=.90+.20*rand;
        fcPowerScale=.90+.10*rand;
        dcdcScale=.975+.025*rand;
        fanPowerScale=.92+.16*rand;
        taskScale=.90+.20*rand;
    end

    % Distinct complete missions: vary climb, horizontal speed, maneuver intensity,
    % initial thermal state and SOC without changing the segment time grid.
    p.mission.vz=p.mission.vz*taskScale.*(.90+.20*rand(size(p.mission.vz)));
    p.mission.vh=p.mission.vh*taskScale.*(.90+.20*rand(size(p.mission.vh)));
    p.mission.maneuver=min(1.5,p.mission.maneuver*taskScale*(.85+.30*rand));
    p.thermal.Tstack0_C=42+6*rand;
    p.thermal.Tair0_C=27+6*rand;
    p.bat.SOC0=.72+.16*rand;
    p.scenario=struct('ambientOffset_C',ambientOffset,'loadScale',loadScale, ...
        'uaScale',uaScale,'capScale',capScale,'fcPowerScale',fcPowerScale, ...
        'dcdcEtaScale',dcdcScale,'fanPowerScale',fanPowerScale);

    controller=controllers(mod(scenarioId-1,numel(controllers))+1);
    [r,~]=run_x650_v03(controller,p,false);
    n=height(r);
    missionPhase=phaseAtTime(r.time_s,p.mission);
    r=[table(repmat(scenarioId,n,1),repmat(split,n,1),missionPhase, ...
        'VariableNames',{'scenario_id','split','mission_phase'}),r];
    allRows=[allRows;r]; %#ok<AGROW>

    row=table(scenarioId,split,controller,p.env.altitudes_m,ambientOffset, ...
        loadScale,uaScale,capScale,fcPowerScale,dcdcScale,fanPowerScale,taskScale, ...
        p.thermal.Tstack0_C,p.thermal.Tair0_C,p.bat.SOC0, ...
        'VariableNames',{'scenario_id','split','controller','altitude_m', ...
        'ambient_offset_C','load_scale','ua_scale','heat_capacity_scale', ...
        'fc_power_scale','dcdc_efficiency_scale','fan_power_scale','task_scale', ...
        'initial_stack_temperature_C','initial_air_temperature_C','initial_SOC'});
    manifest=[manifest;row]; %#ok<AGROW>
end

writetable(allRows,fullfile(out,'thermal_trajectories_v01.csv'));
writetable(manifest,fullfile(out,'scenario_manifest_v01.csv'));
save(fullfile(out,'probability_dataset_v01.mat'),'allRows','manifest','-v7.3');
fprintf('PROBABILITY_DATASET_V01_COMPLETE scenarios=%d rows=%d\n',nScenarios,height(allRows));
end

function value=drawRange(isOod,idRange,oodRange)
if isOod, range=oodRange; else, range=idRange; end
value=range(1)+(range(2)-range(1))*rand;
end

function value=drawSigned(absRange)
value=(absRange(1)+(absRange(2)-absRange(1))*rand);
if rand<.5, value=-value; end
end

function phase=phaseAtTime(t,mission)
phase=zeros(numel(t),1);
for j=1:numel(mission.tStart_s)
    z=t>=mission.tStart_s(j) & t<=mission.tEnd_s(j);
    phase(z)=j;
end
end
