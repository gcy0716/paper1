function check_rl_toolbox_v03()
root=fileparts(mfilename('fullpath')); out=fullfile(root,'results');
if ~exist(out,'dir'), mkdir(out); end
installed=~isempty(ver('reinforcement_learning_toolbox'));
licensed=license('test','RL_Toolbox');
sacClass=exist('rlSACAgent','class')==8 || exist('rlSACAgent','file')==2;
fid=fopen(fullfile(out,'rl_toolbox_check_v03.txt'),'w');
fprintf(fid,'installed=%d\nlicensed=%d\nrlSACAgent_available=%d\n',installed,licensed,sacClass);
fclose(fid);
fprintf('RL_CHECK_COMPLETE installed=%d licensed=%d sac=%d\n',installed,licensed,sacClass);
end
