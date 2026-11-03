function [agent,stats]=finetune_sac_seed_v04(seed,maxEpisodes)
% Transfer fine-tuning from the matching independent v03 seed. This is not
% equivalent to from-scratch v04 training and is reported separately.
if nargin<1, seed=11; end
if nargin<2, maxEpisodes=10; end
root=fileparts(mfilename('fullpath'));
source=fullfile(root,'results','multiseed_v03',sprintf('seed_%d',seed), ...
    sprintf('sac_seed_%d_40episodes_v03.mat',seed));
s=load(source,'agent'); agent=s.agent;
p=parameters_recommended_v04(); rng(seed+4000,'twister');
env=create_pemfc_rl_env_v03("train_v04",seed+4000,p);
seedDir=fullfile(root,'results','multiseed_v04_finetuned',sprintf('seed_%d',seed));
checkpoints=fullfile(seedDir,'checkpoints'); if ~exist(checkpoints,'dir'), mkdir(checkpoints); end
opts=rlTrainingOptions('MaxEpisodes',maxEpisodes,'MaxStepsPerEpisode',1200, ...
    'ScoreAveragingWindowLength',5,'Verbose',true,'Plots','none', ...
    'StopTrainingCriteria','EpisodeCount','StopTrainingValue',maxEpisodes, ...
    'SaveAgentCriteria','EpisodeCount','SaveAgentValue',5,'SaveAgentDirectory',checkpoints);
stats=train(agent,env,opts);
save(fullfile(seedDir,sprintf('sac_seed_%d_finetuned_%depisodes_v04.mat',seed,maxEpisodes)), ...
    'agent','stats','opts','seed','maxEpisodes','p','source');
summary=table(stats.EpisodeIndex,stats.EpisodeReward,stats.AverageReward,stats.EpisodeSteps, ...
    'VariableNames',{'episode','reward','average_reward','steps'});
writetable(summary,fullfile(seedDir,'finetuning_summary.csv'));
fprintf('SEED_V04_FINETUNE_COMPLETE seed=%d episodes=%d\n',seed,maxEpisodes);
end
