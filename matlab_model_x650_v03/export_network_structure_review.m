function export_network_structure_review()
root=fileparts(mfilename('fullpath'));
fid=fopen(fullfile(root,'..','manuscript_ijhe','revision_evidence','saved_network_structure.txt'),'w');
cleanup=onCleanup(@()fclose(fid));
for seed=[11 22 33]
 S=load(fullfile(root,'results','multiseed_v05_finetuned',sprintf('seed_%d',seed),sprintf('sac_seed_%d_finetuned_5episodes_v05.mat',seed)));
 fprintf(fid,'Seed %d\n',seed);
 actor=getActor(S.agent); net=getModel(actor);
 fprintf(fid,'ACTOR\n%s\n',evalc('disp(net.Layers)'));
 fprintf(fid,'CONNECTIONS\n%s\n',evalc('disp(net.Connections)'));
 critics=getCritic(S.agent);
 for j=1:numel(critics)
  net=getModel(critics(j));
  fprintf(fid,'CRITIC %d\n%s\n',j,evalc('disp(net.Layers)'));
  fprintf(fid,'CONNECTIONS\n%s\n',evalc('disp(net.Connections)'));
 end
end
end
