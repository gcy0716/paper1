function evaluate_v04_agents_on_v05()
% Zero-exploration transfer evaluation before computationally expensive
% v05 from-scratch retraining. Results must be labelled transfer-only.
root=fileparts(mfilename('fullpath')); p=parameters_robust_v05();
for seed=[11 22 33]
    file=fullfile('multiseed_v04_finetuned',sprintf('seed_%d',seed), ...
        sprintf('sac_seed_%d_finetuned_10episodes_v04.mat',seed));
    evaluate_sac_pretrain_v03(file,sprintf('v05_transfer_seed%d',seed),p,"test_v04");
end
fprintf('V04_AGENTS_ON_V05_EVALUATION_COMPLETE\n');
end
