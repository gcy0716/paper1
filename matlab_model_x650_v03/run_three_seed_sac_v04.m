function run_three_seed_sac_v04()
for seed=[11 22 33]
    train_sac_seed_v04(seed,40,1200);
end
fprintf('THREE_SEED_V04_TRAINING_COMPLETE\n');
end
