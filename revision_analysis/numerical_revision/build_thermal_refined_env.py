"""Generate a traceable thermal-only substepping variant of the online plant."""
from pathlib import Path
import hashlib
import json
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parents[1]/'matlab_model_x650_v03/create_pemfc_rl_env_v03.m'
old=SOURCE.read_text(encoding='utf-8')
new=old.replace('function env=create_pemfc_rl_env_v03(', 'function env=create_pemfc_thermal_refined(')
block='''logged.Ts=logged.Ts+p.sim.dt*(logged.qgen-Qsa-Qpass)/(p.thermal.Cstack_JpK*logged.capScale);
logged.Ta=logged.Ta+p.sim.dt*(Qsa-Qaa)/p.thermal.Cair_JpK; logged.prevTs=oldTs;'''
replacement='''% Refine thermal propagation only; electrical, control and history clocks remain 0.5 s.
substeps=p.sim.thermalSubsteps;
for thermalStep=1:substeps
    subQsa=p.thermal.UA_stack_air_max_WpK*logged.uaScale*logged.fan^.7*density*(logged.Ts-logged.Ta);
    subQaa=p.thermal.UA_air_amb_max_WpK*logged.uaScale*logged.fan^.7*density*(logged.Ta-logged.Tamb);
    subQpass=p.thermal.UA_passive_WpK*(logged.Ts-logged.Tamb);
    logged.Ts=logged.Ts+(p.sim.dt/substeps)*(logged.qgen-subQsa-subQpass)/(p.thermal.Cstack_JpK*logged.capScale);
    logged.Ta=logged.Ta+(p.sim.dt/substeps)*(subQsa-subQaa)/p.thermal.Cair_JpK;
end
logged.prevTs=oldTs;'''
assert new.count(block)==1
new=new.replace(block,replacement)
new=new.replace("root=fileparts(mfilename('fullpath'));", "root=fileparts(which('create_pemfc_rl_env_v03'));")
target=HERE/'create_pemfc_thermal_refined.m'
target.write_text(new,encoding='utf-8')
(HERE/'thermal_env_provenance.json').write_text(json.dumps({
    'source':str(SOURCE),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'generated_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
    'change':'thermal state propagation substeps only; preserve controller/electrical/history clocks'},indent=2))
