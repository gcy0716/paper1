function rows=export_input_replication_predictions_v01(dataFolder,outputFolder)
% Export per-sample predictions for controller-stratified shortcut auditing.
root=fileparts(mfilename('fullpath')); dataDir=fullfile(root,'results','probability_dataset_v01');
base=fullfile(root,'results','input_ablation_replication_v01');
if nargin>=1, dataDir=dataFolder; end
if nargin<2, outputFolder=base; end
T=readtable(fullfile(dataDir,'thermal_trajectories_v01.csv'),'TextType','string'); T=addDerivedFeatures(T);
rows=table();
for trainingSeed=[20260911 20260912 20260913]
    folder=fullfile(base,sprintf('seed_%d',trainingSeed));
    for variant=["full","minimal"]
        S=load(fullfile(folder,char("quantile_gru_"+variant+"_h10s_v01.mat")));
        for split=["test","ood_test"]
            [X,Y,scenarioIds,controllers]=buildSequences(T,split,S.featureNames,S.mu,S.sigma,S.historySteps,S.horizonSteps,S.strideSteps,S.historyMode);
            P=predictBatched(S.net,X,256)*S.ySigma+S.yMu; p10=P(1,:)-S.calibrationDelta; p50=P(2,:); p90=P(3,:)+S.calibrationDelta;
            n=numel(Y); part=table(repmat(trainingSeed,n,1),repmat(variant,n,1),repmat(split,n,1),scenarioIds,controllers,double(Y),double(p10'),double(p50'),double(p90'), ...
                'VariableNames',{'training_seed','variant','split','scenario_id','controller','actual_temperature_C','p10_C','p50_C','p90_C'});
            rows=[rows;part]; %#ok<AGROW>
        end
    end
end
writetable(rows,fullfile(outputFolder,'input_replication_predictions_v01.csv'));
fprintf('INPUT_REPLICATION_PREDICTIONS_V01_COMPLETE rows=%d\n',height(rows));
end

function T=addDerivedFeatures(T)
T.temperature_rate_Cps=zeros(height(T),1); T.power_rate_Wps=zeros(height(T),1); T.time_fraction=zeros(height(T),1);
ids=unique(T.scenario_id,'stable');
for i=1:numel(ids), z=find(T.scenario_id==ids(i)); [~,o]=sort(T.time_s(z)); z=z(o); T.temperature_rate_Cps(z)=[0;diff(T.stack_temperature_C(z))]/.5; T.power_rate_Wps(z)=[0;diff(T.propulsion_power_W(z))]/.5; T.time_fraction(z)=T.time_s(z)/max(T.time_s(z)); end
end

function [X,Y,scenarioIds,controllers]=buildSequences(T,splitName,featureNames,mu,sigma,historySteps,horizonSteps,strideSteps,historyMode)
ids=unique(T.scenario_id(T.split==splitName),'stable'); nPer=floor((2401-horizonSteps-historySteps)/strideSteps)+1; maxN=max(1,numel(ids)*nPer);
X=zeros(numel(featureNames),historySteps,maxN,'single'); Y=zeros(maxN,1,'single'); scenarioIds=zeros(maxN,1); controllers=strings(maxN,1); cursor=0;
thermalNames={'stack_temperature_C','cathode_air_temperature_C','temperature_rate_Cps','stack_heat_W','stack_to_air_heat_W','air_to_ambient_heat_W'}; thermalIdx=find(ismember(featureNames,thermalNames));
for i=1:numel(ids)
    G=sortrows(T(T.scenario_id==ids(i),:),'time_s'); F=single(G{:,featureNames}); F=(F-mu)./sigma;
    for stop=historySteps:strideSteps:height(G)-horizonSteps
        cursor=cursor+1; seq=F(stop-historySteps+1:stop,:); if historyMode=="current_thermal" && ~isempty(thermalIdx), seq(:,thermalIdx)=repmat(seq(end,thermalIdx),historySteps,1); end
        X(:,:,cursor)=seq'; Y(cursor)=single(G.stack_temperature_C(stop+horizonSteps)); scenarioIds(cursor)=ids(i); controllers(cursor)=G.controller(1);
    end
end
X=X(:,:,1:cursor); Y=Y(1:cursor); scenarioIds=scenarioIds(1:cursor); controllers=controllers(1:cursor);
end

function P=predictBatched(net,X,batchSize)
P=zeros(3,size(X,3),'single');
for start=1:batchSize:size(X,3), ids=start:min(start+batchSize-1,size(X,3)); raw=predict(net,dlarray(X(:,:,ids),'CTB')); q10=raw(1,:); d1=softplusStable(raw(2,:)); d2=softplusStable(raw(3,:)); P(:,ids)=gather(extractdata([q10;q10+d1;q10+d1+d2])); end
end
function y=softplusStable(x), y=log(1+exp(-abs(x)))+max(x,0); end
