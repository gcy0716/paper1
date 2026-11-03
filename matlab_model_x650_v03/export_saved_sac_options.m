function export_saved_sac_options()
root=fileparts(mfilename('fullpath'));
out=fullfile(root,'..','manuscript_ijhe','revision_evidence','saved_sac_options.txt');
fid=fopen(out,'w'); cleaner=onCleanup(@()fclose(fid));
for seed=[11 22 33]
 S=load(fullfile(root,'results','multiseed_v05_finetuned',sprintf('seed_%d',seed),sprintf('sac_seed_%d_finetuned_5episodes_v05.mat',seed)));
 fprintf(fid,'Seed %d\n%s\n',seed,evalc('disp(S.agent.AgentOptions)'));
 fprintf(fid,'Entropy options\n%s\n',evalc('disp(S.agent.AgentOptions.EntropyWeightOptions)'));
end
end
