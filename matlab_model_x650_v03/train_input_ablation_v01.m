function metrics=train_input_ablation_v01(maxEpochs,miniBatchSize,trainingSeed,variantSelection)
% E2 input-source ablation for the 10 s monotone quantile GRU.
% Split, optimiser, hidden size and validation-only model selection are fixed.
if nargin<1, maxEpochs=20; end
if nargin<2, miniBatchSize=128; end
if nargin<3, trainingSeed=20260910; end
if nargin<4, variantSelection=["no_task","no_environment","no_thermal_history","minimal"]; end
root=fileparts(mfilename('fullpath'));
dataDir=fullfile(root,'results','probability_dataset_v01');
if trainingSeed==20260910 && isequal(variantSelection,["no_task","no_environment","no_thermal_history","minimal"])
    out=fullfile(root,'results','input_ablation_v01');
else
    out=fullfile(root,'results','input_ablation_replication_v01',sprintf('seed_%d',trainingSeed));
end
if ~exist(out,'dir'), mkdir(out); end
T=readtable(fullfile(dataDir,'thermal_trajectories_v01.csv'),'TextType','string');
T=addDerivedFeatures(T);

fullFeatures={ ...
    'stack_temperature_C','cathode_air_temperature_C','propulsion_power_W', ...
    'bus_load_W','fc_power_W','battery_power_W','SOC','fan_command', ...
    'fan_power_W','stack_heat_W','stack_to_air_heat_W', ...
    'air_to_ambient_heat_W','altitude_m','ambient_temperature_C', ...
    'air_density_kgpm3','mission_phase','temperature_rate_Cps', ...
    'power_rate_Wps','time_fraction'};
variants=string(variantSelection); featureSets=cell(numel(variants),1); historyModes=strings(size(variants));
for i=1:numel(variants)
    switch variants(i)
        case "full", featureSets{i}=fullFeatures; historyModes(i)="full";
        case "no_task", featureSets{i}=setdiff(fullFeatures,{'mission_phase','time_fraction'},'stable'); historyModes(i)="full";
        case "no_environment", featureSets{i}=setdiff(fullFeatures,{'altitude_m','ambient_temperature_C','air_density_kgpm3'},'stable'); historyModes(i)="full";
        case "no_thermal_history", featureSets{i}=fullFeatures; historyModes(i)="current_thermal";
        case "minimal", featureSets{i}={'stack_temperature_C','fan_command'}; historyModes(i)="full";
        otherwise, error('Unknown input-ablation variant: %s',variants(i));
    end
end

rng(trainingSeed,'twister');
dt=.5; historySteps=round(60/dt); horizonSteps=round(10/dt); strideSteps=10;
metrics=table(); configs=table();
for v=1:numel(variants)
    variant=variants(v); featureNames=featureSets{v}; historyMode=historyModes(v);
    % Resetting to the same seed makes initialisation and batching comparable.
    rng(trainingSeed,'twister');
    trainRows=T.split=="train"; trainMatrix=single(T{trainRows,featureNames});
    mu=mean(trainMatrix,1); sigma=std(trainMatrix,0,1); sigma(sigma<1e-6)=1;
    [XTrain,YTrain]=buildSequences(T,"train",featureNames,mu,sigma,historySteps,horizonSteps,strideSteps,historyMode);
    [XVal,YVal]=buildSequences(T,"validation",featureNames,mu,sigma,historySteps,horizonSteps,strideSteps,historyMode);
    yMu=mean(YTrain,'all'); ySigma=std(YTrain,0,'all'); if ySigma<1e-6, ySigma=1; end
    YTrainNorm=(YTrain-yMu)/ySigma; YValNorm=(YVal-yMu)/ySigma;

    layers=[sequenceInputLayer(numel(featureNames),Normalization="none")
        gruLayer(48,OutputMode="last")
        fullyConnectedLayer(32)
        tanhLayer
        fullyConnectedLayer(3)];
    net=dlnetwork(layers); trailingAvg=[]; trailingAvgSq=[]; iteration=0;
    bestLoss=inf; bestNet=net; patience=6; staleEpochs=0; history=table();
    for epoch=1:maxEpochs
        order=randperm(size(XTrain,3)); epochLoss=0; batches=0;
        for start=1:miniBatchSize:numel(order)
            iteration=iteration+1; ids=order(start:min(start+miniBatchSize-1,numel(order)));
            [loss,gradients]=dlfeval(@modelLoss,net,dlarray(XTrain(:,:,ids),'CTB'),dlarray(YTrainNorm(:,ids),'CB'));
            learnRate=1e-3*.5^(floor((epoch-1)/10));
            [net,trailingAvg,trailingAvgSq]=adamupdate(net,gradients,trailingAvg,trailingAvgSq,iteration,learnRate);
            epochLoss=epochLoss+double(gather(extractdata(loss))); batches=batches+1;
        end
        trainLoss=epochLoss/max(batches,1); valLoss=evaluatePinball(net,XVal,YValNorm,miniBatchSize);
        history=[history;table(variant,epoch,trainLoss,valLoss, ...
            'VariableNames',{'variant','epoch','train_pinball_normalized','validation_pinball_normalized'})]; %#ok<AGROW>
        fprintf('INPUT_ABLATION %s epoch=%d train=%.5f val=%.5f\n',variant,epoch,trainLoss,valLoss);
        if valLoss<bestLoss-1e-5, bestLoss=valLoss; bestNet=net; staleEpochs=0;
        else, staleEpochs=staleEpochs+1; if staleEpochs>=patience, break; end, end
    end
    writetable(history,fullfile(out,char("training_history_"+variant+"_v01.csv")));

    PVal=predictBatched(bestNet,XVal,miniBatchSize)*ySigma+yMu;
    scores=max(PVal(1,:)-YVal,YVal-PVal(3,:)); scores=sort(double(scores));
    k=min(numel(scores),ceil((numel(scores)+1)*.8)); calibrationDelta=max(0,scores(k));
    net=bestNet; %#ok<NASGU>
    save(fullfile(out,char("quantile_gru_"+variant+"_h10s_v01.mat")), ...
        'net','mu','sigma','yMu','ySigma','featureNames','historyMode','historySteps','horizonSteps','strideSteps','bestLoss','calibrationDelta');
    configs=[configs;table(variant,numel(featureNames),string(strjoin(featureNames,'|')),historyMode,bestLoss,calibrationDelta, ...
        'VariableNames',{'variant','n_features','features','history_mode','best_validation_pinball_normalized','calibration_expansion_C'})]; %#ok<AGROW>

    for split=["validation","test","ood_test"]
        [X,Y]=buildSequences(T,split,featureNames,mu,sigma,historySteps,horizonSteps,strideSteps,historyMode);
        P=predictBatched(bestNet,X,miniBatchSize)*ySigma+yMu;
        metrics=[metrics;calculateMetrics(Y,P,variant,split,"raw",numel(featureNames),0)]; %#ok<AGROW>
        Pc=P; Pc(1,:)=Pc(1,:)-calibrationDelta; Pc(3,:)=Pc(3,:)+calibrationDelta;
        metrics=[metrics;calculateMetrics(Y,Pc,variant,split,"validation_conformal",numel(featureNames),calibrationDelta)]; %#ok<AGROW>
    end
end
writetable(configs,fullfile(out,'input_ablation_config_v01.csv'));
writetable(metrics,fullfile(out,'input_ablation_metrics_v01.csv'));
fprintf('INPUT_ABLATION_V01_COMPLETE variants=%d metric_rows=%d\n',numel(variants),height(metrics));
end

function T=addDerivedFeatures(T)
T.temperature_rate_Cps=zeros(height(T),1); T.power_rate_Wps=zeros(height(T),1); T.time_fraction=zeros(height(T),1);
ids=unique(T.scenario_id,'stable');
for i=1:numel(ids)
    z=find(T.scenario_id==ids(i)); [~,order]=sort(T.time_s(z)); z=z(order);
    T.temperature_rate_Cps(z)=[0;diff(T.stack_temperature_C(z))]/.5;
    T.power_rate_Wps(z)=[0;diff(T.propulsion_power_W(z))]/.5;
    T.time_fraction(z)=T.time_s(z)/max(T.time_s(z));
end
end

function [X,Y]=buildSequences(T,splitName,featureNames,mu,sigma,historySteps,horizonSteps,strideSteps,historyMode)
ids=unique(T.scenario_id(T.split==splitName),'stable'); nPer=floor((2401-horizonSteps-historySteps)/strideSteps)+1;
X=zeros(numel(featureNames),historySteps,max(1,numel(ids)*nPer),'single'); Y=zeros(1,size(X,3),'single'); cursor=0;
thermalNames={'stack_temperature_C','cathode_air_temperature_C','temperature_rate_Cps', ...
    'stack_heat_W','stack_to_air_heat_W','air_to_ambient_heat_W'};
thermalIdx=find(ismember(featureNames,thermalNames));
for i=1:numel(ids)
    G=T(T.scenario_id==ids(i),:); G=sortrows(G,'time_s'); F=single(G{:,featureNames}); F=(F-mu)./sigma;
    for stop=historySteps:strideSteps:height(G)-horizonSteps
        cursor=cursor+1; seq=F(stop-historySteps+1:stop,:);
        if historyMode=="current_thermal" && ~isempty(thermalIdx), seq(:,thermalIdx)=repmat(seq(end,thermalIdx),historySteps,1); end
        X(:,:,cursor)=seq'; Y(cursor)=single(G.stack_temperature_C(stop+horizonSteps));
    end
end
X=X(:,:,1:cursor); Y=Y(:,1:cursor);
end

function [loss,gradients]=modelLoss(net,X,Y)
P=orderedQuantiles(forward(net,X)); loss=pinball(Y,P(1,:),.1)+pinball(Y,P(2,:),.5)+pinball(Y,P(3,:),.9);
gradients=dlgradient(loss,net.Learnables);
end
function P=orderedQuantiles(raw)
q10=raw(1,:); d1=softplusStable(raw(2,:)); d2=softplusStable(raw(3,:)); P=[q10;q10+d1;q10+d1+d2];
end
function y=softplusStable(x), y=log(1+exp(-abs(x)))+max(x,0); end
function loss=pinball(Y,P,q), e=Y-P; loss=mean(max(q*e,(q-1)*e),'all'); end
function value=evaluatePinball(net,X,Y,batchSize)
P=predictBatched(net,X,batchSize);
value=double(pinball(Y,P(1,:),.1)+pinball(Y,P(2,:),.5)+pinball(Y,P(3,:),.9));
end
function P=predictBatched(net,X,batchSize)
P=zeros(3,size(X,3),'single');
for start=1:batchSize:size(X,3)
    ids=start:min(start+batchSize-1,size(X,3)); raw=predict(net,dlarray(X(:,:,ids),'CTB'));
    P(:,ids)=gather(extractdata(orderedQuantiles(raw)));
end
end
function row=calculateMetrics(Y,P,variant,splitName,calibrationVariant,nFeatures,delta)
p10=P(1,:); p50=P(2,:); p90=P(3,:); e=double(Y-p50);
coverage=mean(Y>=p10 & Y<=p90); width=mean(p90-p10); event=Y>65; alarm=p90>65;
tp=sum(event&alarm); fp=sum(~event&alarm); fn=sum(event&~alarm);
pb=double(pinball(Y,p10,.1)+pinball(Y,p50,.5)+pinball(Y,p90,.9));
row=table(variant,splitName,calibrationVariant,nFeatures,delta,mean(abs(e)),sqrt(mean(e.^2)),pb,coverage,width,0, ...
    sum(event),tp/max(tp+fn,1),tp/max(tp+fp,1), ...
    'VariableNames',{'variant','split','calibration_variant','n_features','calibration_expansion_C', ...
    'mae_C','rmse_C','pinball_loss_C','picp_p10_p90','mpiw_p10_p90_C','crossing_rate', ...
    'overtemp_events','overtemp_recall','overtemp_precision'});
end
