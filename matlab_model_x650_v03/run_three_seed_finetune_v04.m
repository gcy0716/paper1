function run_three_seed_finetune_v04()
for seed=[11 22 33]
    finetune_sac_seed_v04(seed,10);
end
fprintf('THREE_SEED_V04_FINETUNE_COMPLETE\n');
end
