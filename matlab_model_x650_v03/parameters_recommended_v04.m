function p=parameters_recommended_v04()
% Recommended conceptual architecture from the v03 mass-capacity screen.
% Scaling relations remain assumptions pending component-specific data.
p=parameters_x650_v03();
scale=1500/1100;
p.fc.capacityScale=scale;
p.fc.PratedNet_W=1500;
p.fc.ramp_Wps=p.fc.ramp_Wps*scale;
p.dcdc.Prated_W=2000;
p.bat.Eusable_Wh=22.2*5;
p.mass.fcSystem_kg=1500/1100;
p.mass.bufferBattery_kg=.75;
fixedMass=p.mass.takeoff_kg-1.0-.75;
p.mass.takeoff_kg=fixedMass+p.mass.fcSystem_kg+p.mass.bufferBattery_kg;
p.thermal.Cstack_JpK=p.thermal.Cstack_JpK*scale;
p.thermal.Cair_JpK=p.thermal.Cair_JpK*scale;
p.thermal.UA_stack_air_max_WpK=p.thermal.UA_stack_air_max_WpK*scale^.8;
p.thermal.UA_air_amb_max_WpK=p.thermal.UA_air_amb_max_WpK*scale^.8;
p.fan.Prated_W=p.fan.Prated_W*scale;
end
