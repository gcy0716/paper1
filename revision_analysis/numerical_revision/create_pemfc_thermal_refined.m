function env=create_pemfc_thermal_refined(split,seed,p)
% Function environment for simulation-only safe fan control.
if nargin<1, split="train"; end
if nargin<2, seed=1; end
if nargin<3 || isempty(p), p=parameters_x650_v03(); end
rng(seed,'twister');
obsInfo=rlNumericSpec([8 1],'LowerLimit',-inf(8,1),'UpperLimit',inf(8,1));
obsInfo.Name='normalized_thermal_state';
actInfo=rlNumericSpec([1 1],'LowerLimit',p.fan.minCommand,'UpperLimit',p.fan.maxCommand);
actInfo.Name='fan_command';
env=rlFunctionEnv(obsInfo,actInfo,@(a,l)localStep(a,l,p),@()localReset(p,split));
end

function [obs,logged]=localReset(p,split)
switch string(split)
    case "configured"
        h=p.env.altitudes_m(1); [baseTamb,~]=atmosphere(h,p);
        offset=0; loadScale=1; uaScale=1; capScale=1; fcScale=1; dcdcScale=1; fanPowerScale=1;
        if isfield(p,'scenario')
            if isfield(p.scenario,'ambientOffset_C'), offset=p.scenario.ambientOffset_C; end
            if isfield(p.scenario,'loadScale'), loadScale=p.scenario.loadScale; end
            if isfield(p.scenario,'uaScale'), uaScale=p.scenario.uaScale; end
            if isfield(p.scenario,'capScale'), capScale=p.scenario.capScale; end
            if isfield(p.scenario,'fcPowerScale'), fcScale=p.scenario.fcPowerScale; end
            if isfield(p.scenario,'dcdcEtaScale'), dcdcScale=p.scenario.dcdcEtaScale; end
            if isfield(p.scenario,'fanPowerScale'), fanPowerScale=p.scenario.fanPowerScale; end
        end
    case "train_v04"
        altitudes=[0 1500 3000 4500 5000]; offsets=[-5 0 5];
        h=altitudes(randi(numel(altitudes))); offset=offsets(randi(numel(offsets)));
        loadScale=.90+.18*rand; uaScale=.85+.30*rand; capScale=.90+.20*rand;
        fcScale=.92+.08*rand; dcdcScale=.98+.02*rand; fanPowerScale=.95+.10*rand;
    case "validation_v04"
        h=1000+4500*rand; offset=-5+10*rand;
        loadScale=.93+.17*rand; uaScale=.82+.28*rand; capScale=.87+.23*rand;
        fcScale=.90+.10*rand; dcdcScale=.975+.025*rand; fanPowerScale=.92+.16*rand;
    case "test_v04"
        h=5000+1000*rand; offset=-5+10*rand;
        loadScale=.95+.15*rand; uaScale=.80+.25*rand; capScale=.85+.25*rand;
        fcScale=.88+.12*rand; dcdcScale=.97+.03*rand; fanPowerScale=.90+.20*rand;
    case "train"
        altitudes=[0 1500 3000 4500]; offsets=[-5 0 5];
        h=altitudes(randi(numel(altitudes))); offset=offsets(randi(numel(offsets)));
        loadScale=.95+.10*rand; uaScale=.90+.20*rand; capScale=.90+.20*rand;
        fcScale=1; dcdcScale=1; fanPowerScale=1;
    case "validation"
        choices=[750 2250 3750]; h=choices(randi(3)); offset=-3+7*rand;
        loadScale=.97+.10*rand; uaScale=.90+.14*rand; capScale=.95+.13*rand;
        fcScale=1; dcdcScale=1; fanPowerScale=1;
    otherwise
        choices=[5000 6000]; h=choices(randi(2)); offset=10*rand;
        loadScale=1+0.12*rand; uaScale=.80+.20*rand; capScale=.85+.15*rand;
        fcScale=1; dcdcScale=1; fanPowerScale=1;
end
[Tamb,rho]=atmosphere(h,p); Tamb=Tamb+offset;
initialTs=p.thermal.Tstack0_C; initialTa=p.thermal.Tair0_C; initialSOC=p.bat.SOC0;
logged=struct('k',0,'h',h,'Tamb',Tamb,'rho',rho,'loadScale',loadScale, ...
    'uaScale',uaScale,'capScale',capScale,'fcScale',fcScale,'dcdcScale',dcdcScale, ...
    'fanPowerScale',fanPowerScale,'Ts',initialTs,'Ta',initialTa,'prevTs',initialTs, ...
    'fan',.4,'prevCommand',.4,'soc',initialSOC,'Pfc',0,'qgen',0,'rewardSum',0,'fanWh',0, ...
    'h2g',0,'unmetKJ',0,'violations',0,'overrides',0, ...
    'predictiveOverrides',0,'reactiveOverrides',0,'predictedP90',nan,'riskHistory',zeros(0,19,'single'), ...
    'demandWh',0,'totalLoadWh',0,'servedLoadWh',0,'fcDcWh',0, ...
    'batteryDischargeWh',0,'batteryChargeWh',0, ...
    'temperatureCost',0,'fanCost',0,'smoothCost',0,'unmetCost',0);
obs=observation(logged,p);
end

function [obs,reward,isDone,logged]=localStep(action,logged,p)
proposed=min(max(double(action(1)),p.fan.minCommand),p.fan.maxCommand);
t=logged.k*p.sim.dt; demand=powerDemand(t,logged,p);
[predictedAction,predictiveOverride,predictedP90]=predictiveSafetyFilter(proposed,logged,p);
reactiveEnabled=true;
if isfield(p,'control') && isfield(p.control,'useReactiveSafety'), reactiveEnabled=p.control.useReactiveSafety; end
if reactiveEnabled
    [safeAction,reactiveOverride]=safetyFilter(predictedAction,logged,p);
else
    safeAction=predictedAction; reactiveOverride=false;
end
overridden=predictiveOverride||reactiveOverride;
commandDelta=proposed-logged.prevCommand;
logged.fan=logged.fan+p.sim.dt*(safeAction-logged.fan)/p.fan.tau_s;
logged.fan=min(max(logged.fan,0),1); Pfan=p.fan.Prated_W*logged.fanPowerScale*logged.fan^3;
PfcMax=p.fc.PratedNet_W*max(.70,1-p.fc.altitudePowerLossAt4000*logged.h/4000)*logged.fcScale;
etaDcdc=p.dcdc.eta*logged.dcdcScale;
target=min(max((demand+Pfan)/etaDcdc,p.fc.PminNet_W),PfcMax);
logged.Pfc=logged.Pfc+p.sim.dt*min(max((target-logged.Pfc)/p.fc.tau_s,-p.fc.ramp_Wps),p.fc.ramp_Wps);
ec=pemfc_electrochem_v03(logged.Pfc,logged.Ts,logged.h,p);
logged.Pfc=ec.grossPower_W; logged.qgen=ec.heat_W;
Pdc=min(logged.Pfc*etaDcdc,p.dcdc.Prated_W); req=demand+Pfan-Pdc;
Pbat=min(max(req,-p.bat.PchgMax_W),p.bat.PdisMax_W);
if logged.soc<=p.bat.SOCmin+1e-12 && Pbat>0, Pbat=0; end
if Pbat>=0, ds=-(Pbat/p.bat.etaDis)*p.sim.dt/(p.bat.Eusable_Wh*3600);
else, ds=-Pbat*p.bat.etaChg*p.sim.dt/(p.bat.Eusable_Wh*3600); end
logged.soc=min(max(logged.soc+ds,p.bat.SOCmin),p.bat.SOCmax);
unmet=max(demand+Pfan-Pdc-Pbat,0);
density=(logged.rho/p.env.rho0)^.5;
Qsa=p.thermal.UA_stack_air_max_WpK*logged.uaScale*logged.fan^.7*density*(logged.Ts-logged.Ta);
Qaa=p.thermal.UA_air_amb_max_WpK*logged.uaScale*logged.fan^.7*density*(logged.Ta-logged.Tamb);
Qpass=p.thermal.UA_passive_WpK*(logged.Ts-logged.Tamb);
oldTs=logged.Ts;
% Refine thermal propagation only; electrical, control and history clocks remain 0.5 s.
substeps=p.sim.thermalSubsteps;
for thermalStep=1:substeps
    subQsa=p.thermal.UA_stack_air_max_WpK*logged.uaScale*logged.fan^.7*density*(logged.Ts-logged.Ta);
    subQaa=p.thermal.UA_air_amb_max_WpK*logged.uaScale*logged.fan^.7*density*(logged.Ta-logged.Tamb);
    subQpass=p.thermal.UA_passive_WpK*(logged.Ts-logged.Tamb);
    logged.Ts=logged.Ts+(p.sim.dt/substeps)*(logged.qgen-subQsa-subQpass)/(p.thermal.Cstack_JpK*logged.capScale);
    logged.Ta=logged.Ta+(p.sim.dt/substeps)*(subQsa-subQaa)/p.thermal.Cair_JpK;
end
logged.prevTs=oldTs;
violation=logged.Ts>p.thermal.Tmax_C;
temperatureTerm=((logged.Ts-p.thermal.Tref_C)/10)^2;
fanTerm=Pfan/p.fan.Prated_W; smoothTerm=(commandDelta/.15)^2;
reward=-(p.reward.temperatureWeight*temperatureTerm ...
    +p.reward.fanEnergyWeight*fanTerm ...
    +p.reward.commandSmoothWeight*smoothTerm ...
    +p.reward.violationWeight*violation+p.reward.unmetPowerWeight*unmet);
logged.prevCommand=proposed;
logged.k=logged.k+1; logged.rewardSum=logged.rewardSum+reward;
logged.fanWh=logged.fanWh+Pfan*p.sim.dt/3600;
logged.h2g=logged.h2g+ec.hydrogen_molps*2.01588*p.sim.dt;
logged.unmetKJ=logged.unmetKJ+unmet*p.sim.dt/1000;
logged.demandWh=logged.demandWh+demand*p.sim.dt/3600;
logged.totalLoadWh=logged.totalLoadWh+(demand+Pfan)*p.sim.dt/3600;
logged.servedLoadWh=logged.servedLoadWh+(demand+Pfan-unmet)*p.sim.dt/3600;
logged.fcDcWh=logged.fcDcWh+Pdc*p.sim.dt/3600;
logged.batteryDischargeWh=logged.batteryDischargeWh+max(Pbat,0)*p.sim.dt/3600;
logged.batteryChargeWh=logged.batteryChargeWh+max(-Pbat,0)*p.sim.dt/3600;
logged.violations=logged.violations+violation; logged.overrides=logged.overrides+overridden;
logged.predictiveOverrides=logged.predictiveOverrides+predictiveOverride;
logged.reactiveOverrides=logged.reactiveOverrides+reactiveOverride;
logged.predictedP90=predictedP90;
phase=missionPhase(t,p);
historyRow=single([logged.Ts,logged.Ta,demand,demand+Pfan,logged.Pfc,Pbat,logged.soc, ...
    logged.fan,Pfan,logged.qgen,Qsa,Qaa,logged.h,logged.Tamb,logged.rho,phase, ...
    (logged.Ts-oldTs)/p.sim.dt,0,t/max(p.sim.tEnd,eps)]);
if ~isempty(logged.riskHistory), historyRow(18)=(demand-logged.riskHistory(end,3))/p.sim.dt; end
logged.riskHistory=[logged.riskHistory;historyRow];
if size(logged.riskHistory,1)>120, logged.riskHistory=logged.riskHistory(end-119:end,:); end
logged.temperatureCost=logged.temperatureCost+temperatureTerm;
logged.fanCost=logged.fanCost+fanTerm; logged.smoothCost=logged.smoothCost+smoothTerm;
logged.unmetCost=logged.unmetCost+unmet;
isDone=logged.k>=round(p.sim.tEnd/p.sim.dt) || logged.Ts>p.thermal.Tmax_C+5;
obs=observation(logged,p);
end

function obs=observation(l,p)
rise=(l.Ts-l.prevTs)/p.sim.dt; demand=powerDemand(l.k*p.sim.dt,l,p);
obs=double([(l.Ts-55)/15;(l.Ta-30)/20;rise/.5;demand/1500; ...
    l.h/6000;l.Tamb/40;(l.soc-.25)/.70;l.fan]);
end

function [safe,overridden]=safetyFilter(proposed,l,p)
safe=proposed; overridden=false; Ts=l.Ts; Ta=l.Ta; density=(l.rho/p.env.rho0)^.5;
margin=1;
if isfield(p,'control') && isfield(p.control,'reactiveMargin_C')
    margin=p.control.reactiveMargin_C;
end
if isfield(p,'control') && isfield(p.control,'reactiveMode') && string(p.control.reactiveMode)=="alarm"
    if Ts>=p.thermal.Tmax_C-margin, safe=1; overridden=true; end
    return;
end
for j=1:10
    Qsa=p.thermal.UA_stack_air_max_WpK*l.uaScale*proposed^.7*density*(Ts-Ta);
    Qaa=p.thermal.UA_air_amb_max_WpK*l.uaScale*proposed^.7*density*(Ta-l.Tamb);
    Qpass=p.thermal.UA_passive_WpK*(Ts-l.Tamb);
    Ts=Ts+p.sim.dt*(l.qgen-Qsa-Qpass)/(p.thermal.Cstack_JpK*l.capScale);
    Ta=Ta+p.sim.dt*(Qsa-Qaa)/p.thermal.Cair_JpK;
end
if Ts>=p.thermal.Tmax_C-margin, safe=1; overridden=true; end
end

function [safe,overridden,p90]=predictiveSafetyFilter(proposed,l,p)
safe=proposed; overridden=false; p90=l.predictedP90;
enabled=isfield(p,'control') && isfield(p.control,'usePredictiveSafety') && p.control.usePredictiveSafety;
if ~enabled || size(l.riskHistory,1)<120, return; end
% Update at the locked 5 s interval. Between updates retain no forced action;
% a new prediction is required for each intervention block.
if mod(l.k,10)~=0, return; end
[S,delta]=riskAssets(); F=(l.riskHistory-S.mu)./S.sigma;
raw=predict(S.net,dlarray(reshape(F',[size(F,2),size(F,1),1]),'CTB'));
raw=gather(extractdata(raw)); d1=softplus(raw(2)); d2=softplus(raw(3));
quantile=.9;
if isfield(p.control,'predictiveQuantile'), quantile=p.control.predictiveQuantile; end
if abs(quantile-.5)<1e-9
    % Median output: no conformal interval expansion is applied.
    p90=(raw(1)+d1)*S.ySigma+S.yMu;
elseif abs(quantile-.9)<1e-9
    % Upper quantile plus validation-only conformal expansion.
    applyCalibration=true;
    if isfield(p.control,'predictiveApplyCalibration')
        applyCalibration=p.control.predictiveApplyCalibration;
    end
    p90=(raw(1)+d1+d2)*S.ySigma+S.yMu+double(applyCalibration)*delta;
else
    error('predictiveQuantile must be 0.5 or 0.9');
end
margin=2; if isfield(p.control,'predictiveMargin_C'), margin=p.control.predictiveMargin_C; end
if p90>=p.thermal.Tmax_C-margin, safe=1; overridden=true; end
end

function [S,delta]=riskAssets()
persistent cached cachedDelta
if isempty(cached)
    root=fileparts(which('create_pemfc_rl_env_v03'));
    cached=load(fullfile(root,'results','probability_dataset_v01','quantile_gru_results','quantile_gru_h10s_v01.mat'));
    C=readtable(fullfile(root,'results','probability_dataset_v01','quantile_gru_results','quantile_gru_calibration_v01.csv'));
    cachedDelta=C.interval_expansion_C(C.horizon_s==10);
end
S=cached; delta=cachedDelta;
end

function y=softplus(x), y=log(1+exp(-abs(x)))+max(x,0); end

function P=powerDemand(t,l,p)
[vz,vh,man]=mission(min(t,p.sim.tEnd),p);
Phover=p.x650.referenceHoverPower_W*(p.mass.takeoff_kg/p.x650.referenceMass_kg)^p.power.massExponent*(p.env.rho0/l.rho)^p.power.densityExponent;
P=(Phover+max(vz,0)*p.mass.takeoff_kg*p.env.g*p.power.climbGain+p.power.horizontalCoeff*vh^2)*(1+p.power.maneuverGain*man)+p.power.avionics_W;
P=P*(1+.02*sin(2*pi*t/17))*l.loadScale;
end

function [vz,vh,man]=mission(t,p)
j=find(t>=p.mission.tStart_s & t<=p.mission.tEnd_s,1,'last');
vz=p.mission.vz(j); vh=p.mission.vh(j); man=p.mission.maneuver(j);
end

function phase=missionPhase(t,p)
phase=find(t>=p.mission.tStart_s & t<=p.mission.tEnd_s,1,'last');
if isempty(phase), phase=numel(p.mission.tStart_s); end
end

function [Tamb,rho]=atmosphere(h,p)
Tk=p.env.T0_K-p.env.lapse_Kpm*h;
pressure=p.env.p0_Pa*(Tk/p.env.T0_K)^(p.env.g/(p.env.R*p.env.lapse_Kpm));
rho=pressure/(p.env.R*Tk); Tamb=Tk-273.15;
end
