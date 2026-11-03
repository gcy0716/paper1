function [agent,stats]=train_sac_seed_v04(seed,maxEpisodes,maxSteps)
% SAC retraining for the recommended 1.5 kW + 5 Ah architecture.
if nargin<1, seed=11; end
if nargin<2, maxEpisodes=40; end
if nargin<3, maxSteps=1200; end
root=fileparts(mfilename('fullpath'));
seedDir=fullfile(root,'results','multiseed_v04',sprintf('seed_%d',seed));
checkpointDir=fullfile(seedDir,'checkpoints');
if ~exist(checkpointDir,'dir'), mkdir(checkpointDir); end
p=parameters_recommended_v04();
rng(seed,'twister'); env=create_pemfc_rl_env_v03("train_v04",seed,p);
obsInfo=getObservationInfo(env); actInfo=getActionInfo(env);
agentOpts=rlSACAgentOptions('SampleTime',p.sim.dt,'ExperienceBufferLength',2e5, ...
    'MiniBatchSize',128,'DiscountFactor',0.995);
agent=rlSACAgent(obsInfo,actInfo,agentOpts);
trainOpts=rlTrainingOptions('MaxEpisodes',maxEpisodes,'MaxStepsPerEpisode',maxSteps, ...
    'ScoreAveragingWindowLength',5,'Verbose',true,'Plots','none', ...
    'StopTrainingCriteria','EpisodeCount','StopTrainingValue',maxEpisodes, ...
    'SaveAgentCriteria','EpisodeCount','SaveAgentValue',10, ...
    'SaveAgentDirectory',checkpointDir);
stats=train(agent,env,trainOpts);
save(fullfile(seedDir,sprintf('sac_seed_%d_%depisodes_v04.mat',seed,maxEpisodes)), ...
    'agent','stats','trainOpts','agentOpts','seed','maxEpisodes','p');
summary=table(stats.EpisodeIndex,stats.EpisodeReward,stats.AverageReward,stats.EpisodeSteps, ...
    'VariableNames',{'episode','reward','average_reward','steps'});
writetable(summary,fullfile(seedDir,'training_summary.csv'));
fid=fopen(fullfile(seedDir,'TRAINING_COMPLETE.txt'),'w');
fprintf(fid,'seed=%d\nepisodes=%d\nsteps_per_episode=%d\narchitecture=1500W_5Ah\n',seed,maxEpisodes,maxSteps); fclose(fid);
fprintf('SEED_V04_TRAINING_COMPLETE seed=%d episodes=%d\n',seed,maxEpisodes);
end
