function run_multiseed_sac_v03(seeds)
% Planned reproducibility runner. Full training is intentionally explicit.
if nargin<1, seeds=[11 22 33 44 55]; end
root=fileparts(mfilename('fullpath')); out=fullfile(root,'results','sac_multiseed');
if ~exist(out,'dir'), mkdir(out); end
for seed=seeds
    rng(seed,'twister');
    [agent,stats]=train_sac_v03("full",seed); %#ok<ASGLU>
    save(fullfile(out,sprintf('sac_seed_%d_v03.mat',seed)),'agent','stats','seed');
end
fprintf('SAC_MULTISEED_COMPLETE seeds=%s\n',mat2str(seeds));
end
