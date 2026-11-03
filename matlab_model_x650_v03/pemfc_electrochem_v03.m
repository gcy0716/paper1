function e = pemfc_electrochem_v03(Pcommand_W,Tstack_C,altitude_m,p)
% Control-oriented polarization, hydrogen and heat model.
% The curve is anchored at the public H-1000 rated point. It is not a
% substitute for digitized validation data over the full operating range.
pressureRatio=(1-2.25577e-5*altitude_m)^5.25588;
capacityScale=1;
if isfield(p.fc,'capacityScale'), capacityScale=p.fc.capacityScale; end
Imax=min(p.fc.Ilimit_A*0.98,p.fc.Irated_A*max(.70,1-p.fc.altitudePowerLossAt4000*altitude_m/4000));
temperatureCorrection=2e-4*(Tstack_C-25); % V/cell, ASSUMPTION
voltage=@(I) p.fc.Ncells*(p.fc.Voc_cell_V+temperatureCorrection ...
    -p.fc.activation_V*log(1+I) ...
    -p.fc.ohmic_Ohm*I ...
    -p.fc.concentration_V*(-log(max(1-I/p.fc.Ilimit_A,1e-6))) ...
    +0.5*4.3085e-5*(Tstack_C+273.15)*log(max(pressureRatio,0.1)));
power=@(I) max(voltage(I),0).*I;
PmaxModule=power(Imax); Pmax=capacityScale*PmaxModule;
target=min(max(Pcommand_W,0),Pmax)/capacityScale;
lo=0; hi=Imax;
for j=1:45
    mid=(lo+hi)/2;
    if power(mid)<target, lo=mid; else, hi=mid; end
end
I=(lo+hi)/2; V=voltage(I); Pgross=capacityScale*V*I;
H2_molps=capacityScale*p.fc.Ncells*I/(2*p.fc.F_Cpmol*p.fc.hydrogenUtilization);
Qreaction=capacityScale*max((p.fc.Ncells*p.fc.Vthermoneutral_cell_V-V)*I,0);
e=struct('current_A',capacityScale*I,'voltage_V',V,'grossPower_W',Pgross, ...
    'maxPower_W',Pmax,'hydrogen_molps',H2_molps,'heat_W',Qreaction);
end
