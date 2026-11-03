function [agent,stats]=finetune_sac_seed_v05(seed,maxEpisodes,maxSteps)
% Continue each independently seeded v04 policy on the 1.7 kW v05 system.
if nargin<1, seed=11; end
if nargin<2, maxEpisodes=20; end
if nargin<3, maxSteps=1200; end
root=fileparts(mfilename('fullpath'));
source=fullfile(root,'results','multiseed_v04_finetuned',sprintf('seed_%d',seed), ...
    sprintf('sac_seed_%d_finetuned_10episodes_v04.mat',seed));
s=load(source,'agent'); sourceAgent=s.agent;
% Transfer networks only. Reusing the replay memory would mix obsolete
% architecture transitions into v05 and made updates prohibitively slow.
actor=getActor(sourceAgent); critics=getCritic(sourceAgent);
agentOpts=sourceAgent.AgentOptions;
agentOpts.ExperienceBufferLength=1e5;
agentOpts.MiniBatchSize=64;
agentOpts.NumWarmStartSteps=64;
agentOpts.PolicyUpdateFrequency=20;
agentOpts.CriticUpdateFrequency=20;
agentOpts.TargetUpdateFrequency=20;
agent=rlSACAgent(actor,critics,agentOpts);
p=parameters_robust_v05(); rng(seed+5000,'twister');
env=create_pemfc_rl_env_v03("train_v04",seed+5000,p);
seedDir=fullfile(root,'results','multiseed_v05_finetuned',sprintf('seed_%d',seed));
checkpoints=fullfile(seedDir,'checkpoints'); if ~exist(checkpoints,'dir'), mkdir(checkpoints); end
opts=rlTrainingOptions('MaxEpisodes',maxEpisodes,'MaxStepsPerEpisode',maxSteps, ...
    'ScoreAveragingWindowLength',5,'Verbose',true,'Plots','none', ...
    'StopTrainingCriteria','EpisodeCount','StopTrainingValue',maxEpisodes, ...
    'SaveAgentCriteria','EpisodeCount','SaveAgentValue',5,'SaveAgentDirectory',checkpoints);
stats=train(agent,env,opts);
save(fullfile(seedDir,sprintf('sac_seed_%d_finetuned_%depisodes_v05.mat',seed,maxEpisodes)), ...
    'agent','stats','opts','seed','maxEpisodes','p','source');
summary=table(stats.EpisodeIndex,stats.EpisodeReward,stats.AverageReward,stats.EpisodeSteps, ...
    'VariableNames',{'episode','reward','average_reward','steps'});
writetable(summary,fullfile(seedDir,'finetuning_summary.csv'));
fprintf('SEED_V05_FINETUNE_COMPLETE seed=%d episodes=%d\n',seed,maxEpisodes);
end
