function evaluate_three_seed_sac_v04()
root=fileparts(mfilename('fullpath')); p=parameters_recommended_v04();
for seed=[11 22 33]
    file=fullfile('multiseed_v04',sprintf('seed_%d',seed), ...
        sprintf('sac_seed_%d_40episodes_v04.mat',seed));
    evaluate_sac_pretrain_v03(file,sprintf('v04_seed%d',seed),p,"test_v04");
end
fprintf('THREE_SEED_V04_EVALUATION_COMPLETE\n');
end
