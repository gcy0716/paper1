function metrics = train_quantile_gru_v01(maxEpochs,miniBatchSize,smokeTest)
% Joint monotone P10/P50/P90 GRU forecasts for 10/30/60 s horizons.
% Training/normalisation/checkpoint selection use train/validation only.
if nargin<1, maxEpochs=30; end
if nargin<2, miniBatchSize=128; end
if nargin<3, smokeTest=false; end

root=fileparts(mfilename('fullpath'));
dataDir=fullfile(root,'results','probability_dataset_v01');
out=fullfile(dataDir,'quantile_gru_results');
if ~exist(out,'dir'), mkdir(out); end
source=fullfile(dataDir,'thermal_trajectories_v01.csv');
assert(exist(source,'file')==2,'Run generate_probability_dataset_v01 first.');

rng(20260910,'twister');
T=readtable(source,'TextType','string');
T=addDerivedFeatures(T);
featureNames={ ...
    'stack_temperature_C','cathode_air_temperature_C','propulsion_power_W', ...
    'bus_load_W','fc_power_W','battery_power_W','SOC','fan_command', ...
    'fan_power_W','stack_heat_W','stack_to_air_heat_W', ...
    'air_to_ambient_heat_W','altitude_m','ambient_temperature_C', ...
    'air_density_kgpm3','mission_phase','temperature_rate_Cps', ...
    'power_rate_Wps','time_fraction'};

trainRows=T.split=="train";
trainMatrix=single(T{trainRows,featureNames});
mu=mean(trainMatrix,1); sigma=std(trainMatrix,0,1); sigma(sigma<1e-6)=1;

dt=0.5; historySeconds=60; historySteps=round(historySeconds/dt);
strideSteps=10; horizonsSeconds=[10 30 60];
metrics=table();

for horizonSeconds=horizonsSeconds
    horizonSteps=round(horizonSeconds/dt);
    [XTrain,YTrain]=buildSequences(T,"train",featureNames,mu,sigma,historySteps,horizonSteps,strideSteps);
    [XVal,YVal]=buildSequences(T,"validation",featureNames,mu,sigma,historySteps,horizonSteps,strideSteps);
    if smokeTest
        XTrain=XTrain(:,:,1:min(512,size(XTrain,3))); YTrain=YTrain(:,1:min(512,size(YTrain,2)));
        XVal=XVal(:,:,1:min(256,size(XVal,3))); YVal=YVal(:,1:min(256,size(YVal,2)));
    end
    yMu=mean(YTrain,'all'); ySigma=std(YTrain,0,'all');
    if ySigma<1e-6, ySigma=1; end
    YTrainNorm=(YTrain-yMu)/ySigma; YValNorm=(YVal-yMu)/ySigma;

    layers=[sequenceInputLayer(numel(featureNames),Normalization="none")
        gruLayer(48,OutputMode="last")
        fullyConnectedLayer(32)
        tanhLayer
        fullyConnectedLayer(3)];
    net=dlnetwork(layers);
    trailingAvg=[]; trailingAvgSq=[]; iteration=0;
    bestLoss=inf; bestNet=net; patience=6; staleEpochs=0;
    historyRows=table();

    for epoch=1:maxEpochs
        order=randperm(size(XTrain,3)); epochLoss=0; batches=0;
        for start=1:miniBatchSize:numel(order)
            iteration=iteration+1;
            ids=order(start:min(start+miniBatchSize-1,numel(order)));
            dlX=dlarray(XTrain(:,:,ids),'CTB');
            dlY=dlarray(YTrainNorm(:,ids),'CB');
            [loss,gradients]=dlfeval(@modelLoss,net,dlX,dlY);
            learnRate=1e-3*0.5^(floor((epoch-1)/10));
            [net,trailingAvg,trailingAvgSq]=adamupdate(net,gradients,trailingAvg,trailingAvgSq,iteration,learnRate);
            epochLoss=epochLoss+double(gather(extractdata(loss))); batches=batches+1;
        end
        trainLoss=epochLoss/max(batches,1);
        valLoss=evaluatePinball(net,XVal,YValNorm,miniBatchSize);
        row=table(horizonSeconds,epoch,trainLoss,valLoss, ...
            'VariableNames',{'horizon_s','epoch','train_pinball','validation_pinball'});
        historyRows=[historyRows;row]; %#ok<AGROW>
        fprintf('GRU h=%ds epoch=%d train=%.5f val=%.5f\n',horizonSeconds,epoch,trainLoss,valLoss);
        if valLoss<bestLoss-1e-5
            bestLoss=valLoss; bestNet=net; staleEpochs=0;
        else
            staleEpochs=staleEpochs+1;
            if staleEpochs>=patience, break; end
        end
    end

    writetable(historyRows,fullfile(out,sprintf('training_history_h%ds_v01.csv',horizonSeconds)));
    net=bestNet; %#ok<NASGU>
    save(fullfile(out,sprintf('quantile_gru_h%ds_v01.mat',horizonSeconds)), ...
        'net','mu','sigma','yMu','ySigma','featureNames','historySteps','horizonSteps','strideSteps','bestLoss');

    for split=["validation","test","ood_test"]
        [X,Y]=buildSequences(T,split,featureNames,mu,sigma,historySteps,horizonSteps,strideSteps);
        P=predictBatched(bestNet,X,miniBatchSize)*ySigma+yMu;
        row=calculateMetrics(Y,P,horizonSeconds,split);
        metrics=[metrics;row]; %#ok<AGROW>
    end
end

writetable(metrics,fullfile(out,'quantile_gru_metrics_v01.csv'));
disp(metrics); fprintf('QUANTILE_GRU_V01_COMPLETE\n');
end

function T=addDerivedFeatures(T)
T.temperature_rate_Cps=zeros(height(T),1);
T.power_rate_Wps=zeros(height(T),1);
T.time_fraction=zeros(height(T),1);
ids=unique(T.scenario_id,'stable');
for i=1:numel(ids)
    z=find(T.scenario_id==ids(i));
    [~,order]=sort(T.time_s(z)); z=z(order);
    T.temperature_rate_Cps(z)=[0;diff(T.stack_temperature_C(z))]/0.5;
    T.power_rate_Wps(z)=[0;diff(T.propulsion_power_W(z))]/0.5;
    T.time_fraction(z)=T.time_s(z)/max(T.time_s(z));
end
end

function [X,Y]=buildSequences(T,splitName,featureNames,mu,sigma,historySteps,horizonSteps,strideSteps)
ids=unique(T.scenario_id(T.split==splitName),'stable');
nPerScenario=floor((2401-horizonSteps-historySteps)/strideSteps)+1;
X=zeros(numel(featureNames),historySteps,max(1,numel(ids)*nPerScenario),'single');
Y=zeros(1,size(X,3),'single'); cursor=0;
for i=1:numel(ids)
    G=T(T.scenario_id==ids(i),:); G=sortrows(G,'time_s');
    F=single(G{:,featureNames}); F=(F-mu)./sigma;
    for stop=historySteps:strideSteps:height(G)-horizonSteps
        cursor=cursor+1; X(:,:,cursor)=F(stop-historySteps+1:stop,:)';
        Y(cursor)=single(G.stack_temperature_C(stop+horizonSteps));
    end
end
X=X(:,:,1:cursor); Y=Y(:,1:cursor);
end

function [loss,gradients]=modelLoss(net,X,Y)
raw=forward(net,X); P=orderedQuantiles(raw);
loss=pinball(Y,P(1,:),.1)+pinball(Y,P(2,:),.5)+pinball(Y,P(3,:),.9);
gradients=dlgradient(loss,net.Learnables);
end

function P=orderedQuantiles(raw)
q10=raw(1,:); d1=softplusStable(raw(2,:)); d2=softplusStable(raw(3,:));
P=[q10;q10+d1;q10+d1+d2];
end

function y=softplusStable(x)
y=log(1+exp(-abs(x)))+max(x,0);
end

function loss=pinball(Y,P,q)
e=Y-P; loss=mean(max(q*e,(q-1)*e),'all');
end

function value=evaluatePinball(net,X,Y,batchSize)
P=predictBatched(net,X,batchSize); e1=Y-P(1,:); e2=Y-P(2,:); e3=Y-P(3,:);
value=mean(max(.1*e1,-.9*e1),'all')+mean(max(.5*e2,-.5*e2),'all')+mean(max(.9*e3,-.1*e3),'all');
end

function P=predictBatched(net,X,batchSize)
P=zeros(3,size(X,3),'single');
for start=1:batchSize:size(X,3)
    ids=start:min(start+batchSize-1,size(X,3));
    raw=predict(net,dlarray(X(:,:,ids),'CTB'));
    P(:,ids)=gather(extractdata(orderedQuantiles(raw)));
end
end

function row=calculateMetrics(Y,P,horizonSeconds,splitName)
p10=P(1,:); p50=P(2,:); p90=P(3,:); e=double(Y-p50);
coverage=mean(Y>=p10 & Y<=p90); width=mean(p90-p10);
event=Y>65; alarm=p90>65; tp=sum(event&alarm); fp=sum(~event&alarm); fn=sum(event&~alarm);
recall=tp/max(tp+fn,1); precision=tp/max(tp+fp,1);
row=table(horizonSeconds,splitName,mean(abs(e)),sqrt(mean(e.^2)),coverage,width,0, ...
    sum(event),recall,precision, ...
    'VariableNames',{'horizon_s','split','mae_C','rmse_C','picp_p10_p90', ...
    'mpiw_p10_p90_C','crossing_rate','overtemp_events','overtemp_recall','overtemp_precision'});
end
