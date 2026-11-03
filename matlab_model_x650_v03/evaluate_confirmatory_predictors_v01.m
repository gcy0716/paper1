function rows=evaluate_confirmatory_predictors_v01()
root=fileparts(mfilename('fullpath')); base=fullfile(root,'results','confirmatory_prediction_v01'); modelBase=fullfile(root,'results','input_ablation_replication_v01');
T=readtable(fullfile(base,'confirmatory_trajectories_v01.csv'),'TextType','string'); T=addFeatures(T); rows=table();
for trainingSeed=[20260911 20260912 20260913]
 for variant=["full","minimal"]
  S=load(fullfile(modelBase,sprintf('seed_%d',trainingSeed),char("quantile_gru_"+variant+"_h10s_v01.mat")));
  [X,Y,candidates,controllers]=buildSequences(T,S.featureNames,S.mu,S.sigma,S.historySteps,S.horizonSteps,S.strideSteps,S.historyMode);
  keep=Y>=61 & Y<=67; X=X(:,:,keep); Y=Y(keep); candidates=candidates(keep); controllers=controllers(keep);
  P=predictBatched(S.net,X,256)*S.ySigma+S.yMu; p10=P(1,:)-S.calibrationDelta; p50=P(2,:); p90=P(3,:)+S.calibrationDelta; n=numel(Y);
  rows=[rows;table(repmat(trainingSeed,n,1),repmat(variant,n,1),candidates,controllers,double(Y),double(p10'),double(p50'),double(p90'), ...
   'VariableNames',{'training_seed','variant','candidate','controller','actual_temperature_C','p10_C','p50_C','p90_C'})]; %#ok<AGROW>
 end
end
writetable(rows,fullfile(base,'confirmatory_predictor_outputs_v01.csv')); fprintf('CONFIRMATORY_PREDICTORS_V01_COMPLETE rows=%d\n',height(rows));
end
function T=addFeatures(T)
T.mission_phase=zeros(height(T),1); T.temperature_rate_Cps=zeros(height(T),1); T.power_rate_Wps=zeros(height(T),1); T.time_fraction=zeros(height(T),1);
p=parameters_robust_v05(); keys=unique(T(:,{'candidate','source_controller'}),'rows');
for i=1:height(keys)
 z=find(T.candidate==keys.candidate(i)&T.source_controller==keys.source_controller(i)); [~,o]=sort(T.time_s(z)); z=z(o);
 for j=1:numel(p.mission.tStart_s), q=T.time_s(z)>=p.mission.tStart_s(j)&T.time_s(z)<=p.mission.tEnd_s(j); T.mission_phase(z(q))=j; end
 T.temperature_rate_Cps(z)=[0;diff(T.stack_temperature_C(z))]/.5; T.power_rate_Wps(z)=[0;diff(T.propulsion_power_W(z))]/.5; T.time_fraction(z)=T.time_s(z)/max(T.time_s(z));
end
end
function [X,Y,candidates,controllers]=buildSequences(T,featureNames,mu,sigma,historySteps,horizonSteps,strideSteps,historyMode)
keys=unique(T(:,{'candidate','source_controller'}),'rows'); maxN=height(keys)*230; X=zeros(numel(featureNames),historySteps,maxN,'single'); Y=zeros(maxN,1,'single'); candidates=zeros(maxN,1); controllers=strings(maxN,1); c=0;
thermal={'stack_temperature_C','cathode_air_temperature_C','temperature_rate_Cps','stack_heat_W','stack_to_air_heat_W','air_to_ambient_heat_W'}; ti=find(ismember(featureNames,thermal));
for i=1:height(keys)
 G=sortrows(T(T.candidate==keys.candidate(i)&T.source_controller==keys.source_controller(i),:),'time_s'); F=single(G{:,featureNames}); F=(F-mu)./sigma;
 for stop=historySteps:strideSteps:height(G)-horizonSteps
  c=c+1; seq=F(stop-historySteps+1:stop,:); if historyMode=="current_thermal"&&~isempty(ti), seq(:,ti)=repmat(seq(end,ti),historySteps,1); end
  X(:,:,c)=seq'; Y(c)=G.stack_temperature_C(stop+horizonSteps); candidates(c)=keys.candidate(i); controllers(c)=keys.source_controller(i);
 end
end
X=X(:,:,1:c); Y=Y(1:c); candidates=candidates(1:c); controllers=controllers(1:c);
end
function P=predictBatched(net,X,batchSize)
P=zeros(3,size(X,3),'single'); for start=1:batchSize:size(X,3), ids=start:min(start+batchSize-1,size(X,3)); raw=predict(net,dlarray(X(:,:,ids),'CTB')); q=raw(1,:); d1=sp(raw(2,:)); d2=sp(raw(3,:)); P(:,ids)=gather(extractdata([q;q+d1;q+d1+d2])); end
end
function y=sp(x), y=log(1+exp(-abs(x)))+max(x,0); end
