function evaluate_three_seed_finetuned_v05()
root=fileparts(mfilename('fullpath')); p=parameters_robust_v05();
for seed=[11 22 33]
    file=fullfile('multiseed_v05_finetuned',sprintf('seed_%d',seed), ...
        sprintf('sac_seed_%d_finetuned_5episodes_v05.mat',seed));
    evaluate_sac_pretrain_v03(file,sprintf('v05_pilot_seed%d',seed),p,"test_v04");
end
fprintf('THREE_SEED_V05_FINETUNED_EVALUATION_COMPLETE\n');
end
