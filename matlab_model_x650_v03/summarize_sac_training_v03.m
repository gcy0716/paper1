function summary=summarize_sac_training_v03(runMode)
if nargin<1, runMode="pretrain"; end
root=fileparts(mfilename('fullpath')); out=fullfile(root,'results');
f=fullfile(out,['sac_agent_' char(runMode) '_v03.mat']);
s=load(f,'stats','seed'); stats=s.stats;
summary=table(stats.EpisodeIndex,stats.EpisodeReward,stats.AverageReward, ...
    stats.EpisodeSteps,'VariableNames',{'episode','reward','average_reward','steps'});
writetable(summary,fullfile(out,['sac_training_' char(runMode) '_summary_v03.csv']));
finite=all(isfinite(summary.reward)); completed=height(summary);
lastWindow=mean(summary.reward(max(1,end-4):end));
fid=fopen(fullfile(out,['sac_training_' char(runMode) '_audit_v03.txt']),'w');
fprintf(fid,'run_mode=%s\nseed=%d\nepisodes=%d\nfinite_rewards=%d\nlast5_mean_reward=%.8g\n', ...
    char(runMode),s.seed,completed,finite,lastWindow); fclose(fid);
disp(summary); fprintf('SAC_TRAINING_SUMMARY_COMPLETE\n');
end
