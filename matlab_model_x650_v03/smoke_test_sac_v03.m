function smoke_test_sac_v03()
% Minimal license/API smoke test; this is not the paper training run.
root=fileparts(mfilename('fullpath')); out=fullfile(root,'results');
obsInfo=rlNumericSpec([8 1],'LowerLimit',-inf(8,1),'UpperLimit',inf(8,1));
obsInfo.Name='thermal_state';
actInfo=rlNumericSpec([1 1],'LowerLimit',0.15,'UpperLimit',1.0);
actInfo.Name='fan_command';
env=rlFunctionEnv(obsInfo,actInfo,@localStep,@localReset);
opts=rlSACAgentOptions('SampleTime',0.5);
agent=rlSACAgent(obsInfo,actInfo,opts); %#ok<NASGU>
validateEnvironment(env);
fid=fopen(fullfile(out,'sac_smoke_test_v03.txt'),'w');
fprintf(fid,'SAC_SMOKE_TEST_PASS\nobservation_dimension=8\naction_dimension=1\n'); fclose(fid);
fprintf('SAC_SMOKE_TEST_PASS\n');
end

function [nextObs,reward,isDone,logged]=localStep(action,logged)
logged.step=logged.step+1;
nextObs=logged.observation; nextObs(8)=action;
logged.observation=nextObs; reward=-double(action).^2; isDone=logged.step>=5;
end

function [obs,logged]=localReset()
obs=zeros(8,1); logged=struct('step',0,'observation',obs);
end
