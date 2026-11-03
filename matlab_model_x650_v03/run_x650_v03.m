function [allResults,summaries] = run_x650_v03(controllerMode,p,writeArtifacts)
% Multi-altitude, two-node thermal baseline with independent safety filter.
if nargin<1, controllerMode="safe_pi"; end
if nargin<2 || isempty(p), p=parameters_x650_v03(); end
if nargin<3, writeArtifacts=true; end
controllerMode=string(controllerMode);
assert(any(controllerMode==["fixed","pi","safe_pi","risk_safe_pi","mpc"]),'Unknown controller mode');
root=fileparts(mfilename('fullpath')); out=fullfile(root,'results');
if ~exist(out,'dir'), mkdir(out); end
assert(p.mass.takeoff_kg<=p.x650.takeoffLimit_kg,'X650 mass limit exceeded');
allResults=table(); summaries=table();
for altitude=p.env.altitudes_m
    r=simulate_case(p,altitude,controllerMode); allResults=[allResults;r]; %#ok<AGROW>
    totalLoadWh=sum(r.bus_load_W)*p.sim.dt/3600;
    unmetKJ=sum(r.unmet_power_W)*p.sim.dt/1000;
    adequacy=1-(unmetKJ/3.6)/max(totalLoadWh,eps);
    s=table(controllerMode,altitude,p.mass.takeoff_kg,max(r.bus_load_W),min(r.SOC),r.SOC(end), ...
        max(r.stack_temperature_C),sum(r.fan_power_W)*p.sim.dt/3600, ...
        sum(r.system_loss_W)*p.sim.dt/3600,sum(r.hydrogen_gps)*p.sim.dt,unmetKJ,adequacy,totalLoadWh,sum(r.safety_override), ...
        sum(r.predictive_override),sum(r.reactive_override), ...
        max(abs(r.bus_residual_W)),max(abs(r.thermal_residual_W)), ...
        'VariableNames',{'controller','altitude_m','takeoff_mass_kg','peak_bus_load_W','minimum_SOC','final_SOC', ...
        'peak_stack_temperature_C','fan_energy_Wh','system_loss_energy_Wh','hydrogen_used_g','unmet_energy_kJ', ...
        'mission_energy_adequacy','total_load_energy_Wh','safety_override_steps','predictive_override_steps', ...
        'reactive_override_steps','max_bus_residual_W','max_thermal_residual_W'});
    summaries=[summaries;s]; %#ok<AGROW>
end
tag=char(controllerMode);
if ~writeArtifacts
    return
end
writetable(allResults,fullfile(out,['x650_timeseries_' tag '_v03.csv']));
writetable(summaries,fullfile(out,['x650_summary_' tag '_v03.csv']));
save(fullfile(out,['x650_' tag '_v03.mat']),'allResults','summaries','p');
f=figure('Color','w','Position',[100 100 1100 720]); tiledlayout(2,2);
for altitude=p.env.altitudes_m
    z=allResults.altitude_m==altitude;
    nexttile(1); hold on; plot(allResults.time_s(z)/60,allResults.stack_temperature_C(z),'DisplayName',sprintf('%g m',altitude));
    nexttile(2); hold on; plot(allResults.time_s(z)/60,allResults.fan_command(z),'DisplayName',sprintf('%g m',altitude));
    nexttile(3); hold on; plot(allResults.time_s(z)/60,allResults.SOC(z)*100,'DisplayName',sprintf('%g m',altitude));
    nexttile(4); hold on; plot(allResults.time_s(z)/60,allResults.bus_load_W(z)/1000,'DisplayName',sprintf('%g m',altitude));
end
nexttile(1); yline(p.thermal.Tmax_C,'r--'); ylabel('Stack temperature (C)'); grid on; legend;
nexttile(2); ylabel('Fan command'); grid on; legend;
nexttile(3); ylabel('SOC (%)'); xlabel('Time (min)'); grid on; legend;
nexttile(4); ylabel('Bus load (kW)'); xlabel('Time (min)'); grid on; legend;
exportgraphics(f,fullfile(out,['x650_overview_' tag '_v03.png']),'Resolution',220); close(f);
disp(summaries); fprintf('X650_V03_COMPLETE\n');
end

function r=simulate_case(p,h,controllerMode)
dt=p.sim.dt; t=(0:dt:p.sim.tEnd)'; n=numel(t);
scenario=struct('ambientOffset_C',0,'loadScale',1,'uaScale',1,'capScale',1, ...
    'fcPowerScale',1,'dcdcEtaScale',1,'fanPowerScale',1);
if isfield(p,'scenario')
    fields=fieldnames(p.scenario);
    for ii=1:numel(fields), scenario.(fields{ii})=p.scenario.(fields{ii}); end
end
T_K=p.env.T0_K-p.env.lapse_Kpm*h;
pressure=p.env.p0_Pa*(T_K/p.env.T0_K)^(p.env.g/(p.env.R*p.env.lapse_Kpm));
rho=pressure/(p.env.R*T_K); Tamb=T_K-273.15+scenario.ambientOffset_C;
vz=zeros(n,1); vh=vz; man=vz; missionPhase=vz;
for j=1:numel(p.mission.tStart_s)
    q=t>=p.mission.tStart_s(j)&t<=p.mission.tEnd_s(j);
    vz(q)=p.mission.vz(j); vh(q)=p.mission.vh(j); man(q)=p.mission.maneuver(j); missionPhase(q)=j;
end
Phover=p.x650.referenceHoverPower_W*(p.mass.takeoff_kg/p.x650.referenceMass_kg)^p.power.massExponent*(p.env.rho0/rho)^p.power.densityExponent;
Pprop=(Phover+max(vz,0)*p.mass.takeoff_kg*p.env.g*p.power.climbGain+p.power.horizontalCoeff*vh.^2).*(1+p.power.maneuverGain*man)+p.power.avionics_W;
Pprop=Pprop.*(1+.02*sin(2*pi*t/17))*scenario.loadScale;
Tst=zeros(n,1); Ta=Tst; Tst(1)=p.thermal.Tstack0_C; Ta(1)=p.thermal.Tair0_C;
u=zeros(n,1); u(1)=.4; integ=0; override=false(n,1); predictiveOverride=false(n,1); reactiveOverride=false(n,1); predictedP90=nan(n,1); Pfc=zeros(n,1); SOC=zeros(n,1); SOC(1)=p.bat.SOC0;
Pfan=zeros(n,1); Pload=Pfan; Pbat=Pfan; unmet=Pfan; surplus=Pfan; Vfc=Pfan; Ifc=Pfan; H2gps=Pfan; Qgen=Pfan; Qsa=Pfan; Qaa=Pfan; Qpass=Pfan; thermalResidual=Pfan; busResidual=Pfan; loss=Pfan;
powerFactor=max(.70,1-p.fc.altitudePowerLossAt4000*h/4000); PfcMax=p.fc.PratedNet_W*powerFactor*scenario.fcPowerScale;
riskEnabled=controllerMode=="risk_safe_pi"; riskUpdateSteps=10; lastP90=nan;
if riskEnabled
    riskModel=load(fullfile(fileparts(mfilename('fullpath')),'results','probability_dataset_v01','quantile_gru_results','quantile_gru_h10s_v01.mat'));
    cal=readtable(fullfile(fileparts(mfilename('fullpath')),'results','probability_dataset_v01','quantile_gru_results','quantile_gru_calibration_v01.csv'));
    riskDelta=cal.interval_expansion_C(cal.horizon_s==10);
    predictiveMargin=p.control.safeMargin_C;
    if isfield(p.control,'predictiveMargin_C'), predictiveMargin=p.control.predictiveMargin_C; end
end
for k=1:n-1
    err=Tst(k)-p.thermal.Tref_C; integ=integ+err*dt;
    if controllerMode=="fixed"
        proposed=0.50;
    elseif controllerMode=="mpc"
        if k==1, qPredict=0; else, qPredict=Qgen(k-1); end
        proposed=mpc_fan_command(Tst(k),Ta(k),Tamb,qPredict,rho,u(k),p);
    else
        proposed=min(max(.35+p.control.Kp*err+p.control.Ki*integ,p.fan.minCommand),p.fan.maxCommand);
    end
    rise=0; if k>1, rise=(Tst(k)-Tst(k-1))/dt; end
    reactiveRisk=Tst(k)>=p.thermal.Tmax_C-p.control.safeMargin_C || rise>p.control.maxRise_Cps;
    predictiveRisk=false;
    if riskEnabled && k>riskModel.historySteps && mod(k-1,riskUpdateSteps)==0
        idx=(k-riskModel.historySteps):(k-1);
        F=[Tst(idx),Ta(idx),Pprop(idx),Pload(idx),Pfc(idx),Pbat(idx),SOC(idx),u(idx),Pfan(idx), ...
            Qgen(idx),Qsa(idx),Qaa(idx),repmat(h,numel(idx),1),repmat(Tamb,numel(idx),1), ...
            repmat(rho,numel(idx),1),missionPhase(idx),[0;diff(Tst(idx))]/dt, ...
            [0;diff(Pprop(idx))]/dt,t(idx)/max(t)];
        F=(single(F)-riskModel.mu)./riskModel.sigma;
        raw=predict(riskModel.net,dlarray(reshape(F',[size(F,2),size(F,1),1]),'CTB'));
        raw=gather(extractdata(raw)); d1=log(1+exp(-abs(raw(2))))+max(raw(2),0); d2=log(1+exp(-abs(raw(3))))+max(raw(3),0);
        lastP90=(raw(1)+d1+d2)*riskModel.ySigma+riskModel.yMu+riskDelta;
    end
    if riskEnabled
        predictedP90(k)=lastP90; predictiveRisk=~isnan(lastP90) && lastP90>=p.thermal.Tmax_C-predictiveMargin;
    end
    if (controllerMode=="safe_pi" && reactiveRisk) || (riskEnabled && (predictiveRisk || reactiveRisk))
        proposed=1; override(k)=true;
        predictiveOverride(k)=predictiveRisk; reactiveOverride(k)=reactiveRisk;
    end
    u(k+1)=u(k)+dt*(proposed-u(k))/p.fan.tau_s; u(k+1)=min(max(u(k+1),0),1);
    Pfan(k)=p.fan.Prated_W*scenario.fanPowerScale*u(k)^p.fan.powerExponent; Pload(k)=Pprop(k)+Pfan(k);
    etaDcdc=p.dcdc.eta*scenario.dcdcEtaScale;
    target=min(max(Pload(k)/etaDcdc,p.fc.PminNet_W),PfcMax);
    Pfc(k+1)=Pfc(k)+dt*min(max((target-Pfc(k))/p.fc.tau_s,-p.fc.ramp_Wps),p.fc.ramp_Wps);
    Pdc=min(Pfc(k)*etaDcdc,p.dcdc.Prated_W); req=Pload(k)-Pdc;
    Pbat(k)=min(max(req,-p.bat.PchgMax_W),p.bat.PdisMax_W);
    if SOC(k)<=p.bat.SOCmin+1e-12 && Pbat(k)>0, Pbat(k)=0; end
    if SOC(k)>=p.bat.SOCmax-1e-12 && Pbat(k)<0, Pbat(k)=0; end
    if Pbat(k)>=0, dSOC=-(Pbat(k)/p.bat.etaDis)*dt/(p.bat.Eusable_Wh*3600); else, dSOC=-Pbat(k)*p.bat.etaChg*dt/(p.bat.Eusable_Wh*3600); end
    SOC(k+1)=min(max(SOC(k)+dSOC,p.bat.SOCmin),p.bat.SOCmax);
    unmet(k)=max(Pload(k)-Pdc-Pbat(k),0);
    surplus(k)=max(Pdc+Pbat(k)-Pload(k),0);
    busResidual(k)=Pload(k)+surplus(k)-Pdc-Pbat(k)-unmet(k);
    ec=pemfc_electrochem_v03(Pfc(k),Tst(k),h,p);
    Pfc(k)=ec.grossPower_W; Vfc(k)=ec.voltage_V; Ifc(k)=ec.current_A;
    H2gps(k)=ec.hydrogen_molps*2.01588; Qgen(k)=ec.heat_W;
    densityFan=(rho/p.env.rho0)^.5;
    Qsa(k)=p.thermal.UA_stack_air_max_WpK*scenario.uaScale*u(k)^.7*densityFan*(Tst(k)-Ta(k));
    Qaa(k)=p.thermal.UA_air_amb_max_WpK*scenario.uaScale*u(k)^.7*densityFan*(Ta(k)-Tamb);
    Qpass(k)=p.thermal.UA_passive_WpK*(Tst(k)-Tamb);
    dTs=(Qgen(k)-Qsa(k)-Qpass(k))/(p.thermal.Cstack_JpK*scenario.capScale);
    dTa=(Qsa(k)-Qaa(k))/p.thermal.Cair_JpK;
    Tst(k+1)=Tst(k)+dt*dTs; Ta(k+1)=Ta(k)+dt*dTa;
    thermalResidual(k)=Qgen(k)-Qaa(k)-Qpass(k)-p.thermal.Cstack_JpK*scenario.capScale*dTs-p.thermal.Cair_JpK*dTa;
    loss(k)=Qgen(k)+Pfan(k)+(Pfc(k)-Pdc);
end

Pfan(end)=Pfan(end-1); Pload(end)=Pprop(end)+Pfan(end); Pbat(end)=Pbat(end-1); PdcEnd=min(Pfc(end)*p.dcdc.eta*scenario.dcdcEtaScale,p.dcdc.Prated_W); unmet(end)=max(Pload(end)-PdcEnd-Pbat(end),0); surplus(end)=max(PdcEnd+Pbat(end)-Pload(end),0); Vfc(end)=Vfc(end-1); Ifc(end)=Ifc(end-1); H2gps(end)=H2gps(end-1); Qgen(end)=Qgen(end-1); Qsa(end)=Qsa(end-1); Qaa(end)=Qaa(end-1); Qpass(end)=Qpass(end-1); loss(end)=loss(end-1); busResidual(end)=Pload(end)+surplus(end)-PdcEnd-Pbat(end)-unmet(end);
r=table(repmat(controllerMode,n,1),repmat(h,n,1),t,repmat(Tamb,n,1),repmat(rho,n,1),Pprop,Pload,Pfc,Vfc,Ifc,H2gps,Pbat,unmet,surplus,SOC,u,Pfan,Tst,Ta,Qgen,Qsa,Qaa,Qpass,predictedP90,predictiveOverride,reactiveOverride,override,loss,busResidual,thermalResidual, ...
 'VariableNames',{'controller','altitude_m','time_s','ambient_temperature_C','air_density_kgpm3','propulsion_power_W','bus_load_W','fc_power_W','fc_voltage_V','fc_current_A','hydrogen_gps','battery_power_W','unmet_power_W','surplus_power_W','SOC','fan_command','fan_power_W','stack_temperature_C','cathode_air_temperature_C','stack_heat_W','stack_to_air_heat_W','air_to_ambient_heat_W','passive_heat_W','predicted_p90_temperature_C','predictive_override','reactive_override','safety_override','system_loss_W','bus_residual_W','thermal_residual_W'});
end

function best=mpc_fan_command(Ts0,Ta0,Tamb,Qgen,rho,uPrev,p)
% Receding-horizon nonlinear grid-search MPC. Each candidate holds a
% constant command over the prediction horizon; only the first move is used.
best=p.fan.maxCommand; bestCost=inf; densityFan=(rho/p.env.rho0)^.5;
uaScale=1; capScale=1;
if isfield(p,'scenario')
    if isfield(p.scenario,'uaScale'), uaScale=p.scenario.uaScale; end
    if isfield(p.scenario,'capScale'), capScale=p.scenario.capScale; end
end
for candidate=p.mpc.commandGrid
    Ts=Ts0; Ta=Ta0; cost=p.mpc.moveWeight*(candidate-uPrev)^2;
    for j=1:p.mpc.horizonSteps
        usa=p.thermal.UA_stack_air_max_WpK*uaScale*candidate^.7*densityFan;
        uaa=p.thermal.UA_air_amb_max_WpK*uaScale*candidate^.7*densityFan;
        qsa=usa*(Ts-Ta); qaa=uaa*(Ta-Tamb);
        qpass=p.thermal.UA_passive_WpK*(Ts-Tamb);
        Ts=Ts+p.sim.dt*(Qgen-qsa-qpass)/(p.thermal.Cstack_JpK*capScale);
        Ta=Ta+p.sim.dt*(qsa-qaa)/p.thermal.Cair_JpK;
        cost=cost+((Ts-p.thermal.Tref_C)/p.mpc.temperatureScale_C)^2 ...
            +p.mpc.fanEnergyWeight*candidate^3;
        if Ts>p.thermal.Tmax_C
            cost=cost+p.mpc.constraintPenalty*(Ts-p.thermal.Tmax_C)^2;
        end
    end
    if cost<bestCost, bestCost=cost; best=candidate; end
end
end
