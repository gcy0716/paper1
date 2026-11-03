function evaluate_three_seed_sac_v03()
seeds=[11 22 33];
for seed=seeds
    file=fullfile('multiseed_v03',sprintf('seed_%d',seed), ...
        sprintf('sac_seed_%d_40episodes_v03.mat',seed));
    evaluate_sac_pretrain_v03(file,sprintf('seed%d',seed));
end
fprintf('THREE_SEED_EVALUATION_COMPLETE\n');
end
