function run_three_seed_sac_v03()
seeds=[11 22 33];
for seed=seeds
    fprintf('SEED_TRAINING_START seed=%d\n',seed);
    [agent,stats]=train_sac_seed_v03(seed,40); %#ok<ASGLU>
    clear agent stats
end
root=fileparts(mfilename('fullpath'));
fid=fopen(fullfile(root,'results','multiseed_v03','ALL_THREE_SEEDS_COMPLETE.txt'),'w');
fprintf(fid,'seeds=11,22,33\nepisodes_per_seed=40\n'); fclose(fid);
fprintf('ALL_THREE_SEEDS_COMPLETE\n');
end
