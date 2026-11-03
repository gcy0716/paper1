function benchmark_predictor_inference_v01()
% CPU-side timing of the frozen 10 s quantile GRU. No training or tuning.
root=fileparts(mfilename('fullpath'));
modelDir=fullfile(root,'results','probability_dataset_v01','quantile_gru_results');
outDir=fullfile(root,'results','computational_cost_v01');
if ~exist(outDir,'dir'), mkdir(outDir); end

S=load(fullfile(modelDir,'quantile_gru_h10s_v01.mat'));
C=readtable(fullfile(modelDir,'quantile_gru_calibration_v01.csv'));
delta=C.interval_expansion_C(C.horizon_s==10);

% A normalized 19-variable by 120-step tensor has the exact online shape.
X=zeros(numel(S.featureNames),S.historySteps,1,'single');
history=single(repmat(S.mu,S.historySteps,1));

% Warm-up excludes one-time graph initialisation from steady-state timing.
for k=1:20
    forwardOnly(S.net,X);
    riskInference(S,history,delta);
end

% Repeated blocks expose timing dispersion without retraining or changing the model.
nBlocks=10; iterationsPerBlock=500;
tForward=zeros(nBlocks,1); tEndToEnd=zeros(nBlocks,1);
for b=1:nBlocks
    tic
    for k=1:iterationsPerBlock, y=forwardOnly(S.net,X); end %#ok<NASGU>
    tForward(b)=toc/iterationsPerBlock;
    tic
    for k=1:iterationsPerBlock, y=riskInference(S,history,delta); end %#ok<NASGU>
    tEndToEnd(b)=toc/iterationsPerBlock;
end
controlPeriod_s=5;

metric=["network_forward";"end_to_end_p90"];
allTimes=[tForward,tEndToEnd];
median_time_ms=1000*median(allTimes,1)';
q25_time_ms=1000*prctile(allTimes,25,1)';
q75_time_ms=1000*prctile(allTimes,75,1)';
fraction_of_5s_period=median_time_ms/(1000*controlPeriod_s);
input_features=repmat(numel(S.featureNames),2,1);
history_steps=repmat(S.historySteps,2,1);
history_seconds=repmat(S.historySteps*.5,2,1);
blocks=repmat(nBlocks,2,1); iterations_per_block=repmat(iterationsPerBlock,2,1);
T=table(metric,median_time_ms,q25_time_ms,q75_time_ms,fraction_of_5s_period, ...
    input_features,history_steps,history_seconds,blocks,iterations_per_block);
writetable(T,fullfile(outDir,'predictor_inference_timing_v01.csv'));

fid=fopen(fullfile(outDir,'predictor_inference_environment_v01.txt'),'w');
fprintf(fid,'MATLAB=%s\n',version);
fprintf(fid,'computer=%s\n',computer);
fprintf(fid,'control_period_s=%.6g\n',controlPeriod_s);
fprintf(fid,'timing_method=10_blocks_x_500_iterations_after_20_warmups\n');
fprintf(fid,'device=default_MATLAB_execution_device\n');
fclose(fid);

disp(T)
fprintf('PREDICTOR_INFERENCE_BENCHMARK_V01_COMPLETE\n');
end

function y=forwardOnly(net,X)
raw=predict(net,dlarray(X,'CTB'));
y=gather(extractdata(raw));
end

function p90=riskInference(S,history,delta)
F=(history-S.mu)./S.sigma;
raw=predict(S.net,dlarray(reshape(F',[size(F,2),size(F,1),1]),'CTB'));
raw=gather(extractdata(raw));
d1=softplus(raw(2)); d2=softplus(raw(3));
p90=(raw(1)+d1+d2)*S.ySigma+S.yMu+delta;
end

function y=softplus(x)
y=log(1+exp(-abs(x)))+max(x,0);
end
