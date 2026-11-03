function [agent,stats]=train_sac_seed_v03(seed,maxEpisodes)
% Reproducible from-scratch SAC training for one independent seed.
if nargin<1, seed=11; end
if nargin<2, maxEpisodes=40; end
root=fileparts(mfilename('fullpath'));
seedDir=fullfile(root,'results','multiseed_v03',sprintf('seed_%d',seed));
checkpointDir=fullfile(seedDir,'checkpoints');
if ~exist(checkpointDir,'dir'), mkdir(checkpointDir); end
rng(seed,'twister'); env=create_pemfc_rl_env_v03("train",seed);
obsInfo=getObservationInfo(env); actInfo=getActionInfo(env);
agentOpts=rlSACAgentOptions('SampleTime',0.5,'ExperienceBufferLength',2e5, ...
    'MiniBatchSize',128,'DiscountFactor',0.995);
agent=rlSACAgent(obsInfo,actInfo,agentOpts);
trainOpts=rlTrainingOptions('MaxEpisodes',maxEpisodes,'MaxStepsPerEpisode',1200, ...
    'ScoreAveragingWindowLength',5,'Verbose',true,'Plots','none', ...
    'StopTrainingCriteria','EpisodeCount','StopTrainingValue',maxEpisodes, ...
    'SaveAgentCriteria','EpisodeCount','SaveAgentValue',10, ...
    'SaveAgentDirectory',checkpointDir);
stats=train(agent,env,trainOpts);
save(fullfile(seedDir,sprintf('sac_seed_%d_%depisodes_v03.mat',seed,maxEpisodes)), ...
    'agent','stats','trainOpts','agentOpts','seed','maxEpisodes');
summary=table(stats.EpisodeIndex,stats.EpisodeReward,stats.AverageReward,stats.EpisodeSteps, ...
    'VariableNames',{'episode','reward','average_reward','steps'});
writetable(summary,fullfile(seedDir,'training_summary.csv'));
fid=fopen(fullfile(seedDir,'TRAINING_COMPLETE.txt'),'w');
fprintf(fid,'seed=%d\nepisodes=%d\nsteps_per_episode=1200\n',seed,maxEpisodes); fclose(fid);
fprintf('SEED_TRAINING_COMPLETE seed=%d episodes=%d\n',seed,maxEpisodes);
end
