function [agent,stats]=finetune_sac_v03(seed)
% Continue from the 10-episode pretrain using the current reward candidate.
if nargin<1, seed=42; end
root=fileparts(mfilename('fullpath')); out=fullfile(root,'results');
source=fullfile(out,'sac_agent_pretrain_v03.mat');
s=load(source,'agent'); agent=s.agent; agent.UseExplorationPolicy=true;
rng(seed,'twister'); env=create_pemfc_rl_env_v03("train",seed);
checkpointDir=fullfile(out,'finetune_checkpoints_v03');
if ~exist(checkpointDir,'dir'), mkdir(checkpointDir); end
opts=rlTrainingOptions('MaxEpisodes',15,'MaxStepsPerEpisode',1200, ...
    'ScoreAveragingWindowLength',5,'Verbose',true,'Plots','none', ...
    'StopTrainingCriteria','EpisodeCount','StopTrainingValue',15, ...
    'SaveAgentCriteria','EpisodeCount','SaveAgentValue',5, ...
    'SaveAgentDirectory',checkpointDir);
stats=train(agent,env,opts);
save(fullfile(out,'sac_agent_finetuned_v03.mat'),'agent','stats','opts','seed');
summary=table(stats.EpisodeIndex,stats.EpisodeReward,stats.AverageReward,stats.EpisodeSteps, ...
    'VariableNames',{'episode','reward','average_reward','steps'});
writetable(summary,fullfile(out,'sac_training_finetuned_summary_v03.csv'));
fprintf('SAC_FINETUNE_COMPLETE episodes=%d\n',height(summary));
end
