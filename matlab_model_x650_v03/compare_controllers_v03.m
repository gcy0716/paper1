function comparison = compare_controllers_v03()
% Run all implemented controllers under identical missions and constraints.
root=fileparts(mfilename('fullpath')); out=fullfile(root,'results');
modes=["fixed","pi","safe_pi","mpc"]; comparison=table();
for mode=modes
    run_x650_v03(mode);
    s=readtable(fullfile(out,['x650_summary_' char(mode) '_v03.csv']), ...
        'VariableNamingRule','preserve');
    comparison=[comparison;s]; %#ok<AGROW>
end
writetable(comparison,fullfile(out,'controller_comparison_v03.csv'));
disp(comparison);
fprintf('CONTROLLER_COMPARISON_V03_COMPLETE\n');
end
