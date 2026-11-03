function p = parameters_x650_v03()
% Traceable simulation-first baseline. Values tagged ASSUMPTION must be
% included in sensitivity analysis and must not be described as measured.

p.sim.dt = 0.5; p.sim.tEnd = 1200;
p.env.altitudes_m = [0 3000 6000];
p.env.T0_K = 288.15; p.env.p0_Pa = 101325; p.env.rho0 = 1.225;
p.env.lapse_Kpm = 0.0065; p.env.g = 9.80665; p.env.R = 287.05;

% Holybro X650 public platform data.
p.x650.kitMass_kg = 2.0; p.x650.takeoffLimit_kg = 6.3;
p.x650.referenceMass_kg = 3.2; p.x650.referenceEnergy_Wh = 22.2*10*0.85;
p.x650.referenceHoverTime_h = 0.5;
p.x650.referenceHoverPower_W = p.x650.referenceEnergy_Wh/p.x650.referenceHoverTime_h;

% Conceptual retrofit mass budget (ASSUMPTION).
p.mass.fcSystem_kg = 1.00; p.mass.tankRegulatorH2_kg = 0.85;
p.mass.dcdc_kg = 0.25; p.mass.bufferBattery_kg = 0.75;
p.mass.mountWiring_kg = 0.35;
p.mass.takeoff_kg = p.x650.kitMass_kg+p.mass.fcSystem_kg+ ...
    p.mass.tankRegulatorH2_kg+p.mass.dcdc_kg+p.mass.bufferBattery_kg+p.mass.mountWiring_kg;

% Propulsion grey-box model (ASSUMPTION except X650 calibration point).
p.power.massExponent = 1.5; p.power.densityExponent = 0.5;
p.power.climbGain = 1.0; p.power.horizontalCoeff = 2.0;
p.power.maneuverGain = 0.30; p.power.avionics_W = 35;

% Electrical/thermal reference: Donateo et al. open 1 kW dynamic dataset:
% 50 cells, 50 cm2, peak 1100 W, 0--33.5 A @ 30 V, 4.9 kg, Tmax 65 C.
% This reference hardware is NOT compatible with the X650 mass budget.
p.fc.PratedNet_W = 1100; p.fc.PminNet_W = 150;
p.fc.tau_s = 12; p.fc.ramp_Wps = 100;              % ASSUMPTION
p.fc.Ncells = 50; p.fc.activeArea_cm2 = 50;
p.fc.Irated_A = 33.5; p.fc.Vrated_V = 30; p.fc.Ilimit_A = 42;
p.fc.Vthermoneutral_cell_V = 1.254;
p.fc.hydrogenUtilization = 0.95; p.fc.F_Cpmol = 96485.33212;
% Control-oriented polarization coefficients. R is fitted so the curve
% passes through the public rated point; a, b and open-circuit voltage are
% assumptions until digitized polarization data are available.
p.fc.Voc_cell_V = 0.89118949; p.fc.activation_V = 0.06028468;
p.fc.ohmic_Ohm = 0.00098430; p.fc.concentration_V = 0.030;
p.fc.altitudePowerLossAt4000 = 0.056; % midpoint of published 4.7--6.5% range
p.fc.referenceHardwareMass_kg = 4.9; % open-paper Table 3

p.dcdc.eta = 0.94; p.dcdc.Prated_W = 1200; p.dcdc.Vbus_V = 24;
p.bat.Eusable_Wh = 22.2*5; p.bat.SOC0 = 0.80; p.bat.SOCmin = 0.25;
p.bat.SOCmax = 0.95; p.bat.PdisMax_W = 2500; p.bat.PchgMax_W = 400;
p.bat.etaDis = 0.95; p.bat.etaChg = 0.92;

% Two-node stack/cathode-air thermal model (ASSUMPTION pending calibration).
p.thermal.Cstack_JpK = 750; % published calibrated total stack heat capacity
p.thermal.Cair_JpK = 900;   % ASSUMPTION for added second node
p.thermal.UA_stack_air_max_WpK = 50; p.thermal.UA_air_amb_max_WpK = 45;
p.thermal.UA_passive_WpK = 4; p.thermal.Tstack0_C = 45;
p.thermal.Tair0_C = 30; p.thermal.Tref_C = 55; p.thermal.Tmax_C = 65;
p.fan.Prated_W = 45; p.fan.tau_s = 3; p.fan.powerExponent = 3;
p.fan.minCommand = 0.15; p.fan.maxCommand = 1.0;

p.control.Kp = 0.045; p.control.Ki = 0.0012; % baseline PI, ASSUMPTION
p.control.safeMargin_C = 2; p.control.maxRise_Cps = 0.20;
p.mpc.horizonSteps = 30; p.mpc.commandGrid = 0.15:0.05:1.0;
p.mpc.temperatureScale_C = 5; p.mpc.fanEnergyWeight = 0.08;
p.mpc.moveWeight = 0.20; p.mpc.constraintPenalty = 1e5;

% RL reward candidate weights (must be included in sensitivity analysis).
p.reward.temperatureWeight = 1.0;
p.reward.fanEnergyWeight = 0.12;
p.reward.commandSmoothWeight = 0.08;
p.reward.violationWeight = 100;
p.reward.unmetPowerWeight = 0.002;

p.mission.tStart_s = [0 30 120 360 660 780 1080];
p.mission.tEnd_s = [30 120 360 660 780 1080 1200];
p.mission.vz = [0.8 2 0 0 0.5 0 -1.2];
p.mission.vh = [0 0 0 8 5 0 0]; p.mission.maneuver = [0 0 0 0 1 0 0];
end
