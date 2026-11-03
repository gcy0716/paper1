function metrics=calibrate_quantile_gru_v01()
% Validation-only conformal expansion of trained GRU P10-P90 intervals.
root=fileparts(mfilename('fullpath'));
dataDir=fullfile(root,'results','probability_dataset_v01');
modelDir=fullfile(dataDir,'quantile_gru_results');
T=readtable(fullfile(dataDir,'thermal_trajectories_v01.csv'),'TextType','string');
T=addDerived(T); horizons=[10 30 60]; dt=.5; targetCoverage=.8;
metrics=table(); calibration=table();

for horizon=horizons
    S=load(fullfile(modelDir,sprintf('quantile_gru_h%ds_v01.mat',horizon)));
    [Xv,Yv]=buildSeq(T,"validation",S.featureNames,S.mu,S.sigma,S.historySteps,S.horizonSteps,S.strideSteps);
    Pv=predictBatch(S.net,Xv,256)*S.ySigma+S.yMu;
    scores=max(Pv(1,:)-Yv,Yv-Pv(3,:)); scores=sort(double(scores));
    k=min(numel(scores),ceil((numel(scores)+1)*targetCoverage));
    delta=max(0,scores(k));
    calibration=[calibration;table(horizon,targetCoverage,delta,numel(scores), ...
        'VariableNames',{'horizon_s','target_coverage','interval_expansion_C','validation_samples'})]; %#ok<AGROW>
    for split=["validation","test","ood_test"]
        [X,Y]=buildSeq(T,split,S.featureNames,S.mu,S.sigma,S.historySteps,S.horizonSteps,S.strideSteps);
        P=predictBatch(S.net,X,256)*S.ySigma+S.yMu;
        raw=calc(Y,P,horizon,split,"raw");
        Pc=P; Pc(1,:)=Pc(1,:)-delta; Pc(3,:)=Pc(3,:)+delta;
        calibrated=calc(Y,Pc,horizon,split,"validation_conformal");
        metrics=[metrics;raw;calibrated]; %#ok<AGROW>
    end
end
writetable(calibration,fullfile(modelDir,'quantile_gru_calibration_v01.csv'));
writetable(metrics,fullfile(modelDir,'quantile_gru_calibrated_metrics_v01.csv'));
disp(calibration); disp(metrics); fprintf('QUANTILE_GRU_CALIBRATION_V01_COMPLETE\n');
end

function T=addDerived(T)
T.temperature_rate_Cps=zeros(height(T),1); T.power_rate_Wps=zeros(height(T),1); T.time_fraction=zeros(height(T),1);
ids=unique(T.scenario_id,'stable');
for i=1:numel(ids)
    z=find(T.scenario_id==ids(i)); [~,o]=sort(T.time_s(z)); z=z(o);
    T.temperature_rate_Cps(z)=[0;diff(T.stack_temperature_C(z))]/.5;
    T.power_rate_Wps(z)=[0;diff(T.propulsion_power_W(z))]/.5;
    T.time_fraction(z)=T.time_s(z)/max(T.time_s(z));
end
end

function [X,Y]=buildSeq(T,splitName,names,mu,sigma,histSteps,horizonSteps,stride)
ids=unique(T.scenario_id(T.split==splitName),'stable'); X=[]; Y=[];
for i=1:numel(ids)
    G=sortrows(T(T.scenario_id==ids(i),:),'time_s'); F=single(G{:,names}); F=(F-mu)./sigma;
    stops=histSteps:stride:height(G)-horizonSteps;
    Xi=zeros(numel(names),histSteps,numel(stops),'single'); Yi=zeros(1,numel(stops),'single');
    for j=1:numel(stops)
        s=stops(j); Xi(:,:,j)=F(s-histSteps+1:s,:)'; Yi(j)=single(G.stack_temperature_C(s+horizonSteps));
    end
    X=cat(3,X,Xi); Y=[Y Yi]; %#ok<AGROW>
end
end

function P=predictBatch(net,X,batch)
P=zeros(3,size(X,3),'single');
for s=1:batch:size(X,3)
    ids=s:min(s+batch-1,size(X,3)); raw=predict(net,dlarray(X(:,:,ids),'CTB'));
    raw=gather(extractdata(raw)); d1=softplus(raw(2,:)); d2=softplus(raw(3,:));
    P(:,ids)=[raw(1,:);raw(1,:)+d1;raw(1,:)+d1+d2];
end
end

function y=softplus(x), y=log(1+exp(-abs(x)))+max(x,0); end

function row=calc(Y,P,horizon,splitName,variant)
p10=P(1,:); p50=P(2,:); p90=P(3,:); e=double(Y-p50); event=Y>65; alarm=p90>65;
tp=sum(event&alarm); fp=sum(~event&alarm); fn=sum(event&~alarm);
row=table(horizon,splitName,variant,mean(abs(e)),sqrt(mean(e.^2)), ...
    mean(Y>=p10&Y<=p90),mean(p90-p10),sum(event),tp/max(tp+fn,1),tp/max(tp+fp,1), ...
    'VariableNames',{'horizon_s','split','variant','mae_C','rmse_C','picp_p10_p90', ...
    'mpiw_p10_p90_C','overtemp_events','overtemp_recall','overtemp_precision'});
end
