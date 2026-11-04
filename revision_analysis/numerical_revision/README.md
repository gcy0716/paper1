# Additional numerical revision checks

These are exploratory model-in-the-loop checks added during revision. They do
not supply measured thermal/airflow validation or a real flight trajectory.
Original controller weights and locked-test records are unchanged.

## Dependencies and execution

MATLAB R2023b with Reinforcement Learning Toolbox and Deep Learning Toolbox;
Python 3 with NumPy, pandas, SciPy and matplotlib. Work from the repository root.
The commands below rerun the analyses and overwrite their corresponding retained
CSV files. Preserve the supplied result files if comparing a new environment.

```text
python revision_analysis/numerical_revision/build_thermal_refined_env.py
```

In MATLAB, run:

```matlab
addpath('revision_analysis/numerical_revision')
run_numerical_revision('benchmark')
run_numerical_revision('thermal')
run_numerical_revision('failure_diagnostics')
run_numerical_revision('refinement')
run_numerical_revision('rating')
run_numerical_revision('manifests')
run_numerical_revision('development')
```

Then, in Python:

```text
python revision_analysis/numerical_revision/analyze_numerical_revision.py benchmark
python revision_analysis/numerical_revision/analyze_numerical_revision.py thermal
python revision_analysis/numerical_revision/analyze_numerical_revision.py refinement
python revision_analysis/numerical_revision/analyze_numerical_revision.py select
```

Only after the development selection has been frozen, in MATLAB:

```matlab
run_numerical_revision('holdout')
run_numerical_revision('refined_holdout')
```

Then:

```text
python revision_analysis/numerical_revision/analyze_numerical_revision.py holdout
python revision_analysis/numerical_revision/analyze_numerical_revision.py refined_holdout
python revision_analysis/numerical_revision/plot_additional_checks.py thermal
python revision_analysis/numerical_revision/plot_additional_checks.py fair
```

## Interpretation boundaries

- Thermal replay uses exact linear updates with recorded heat generation and fan
  states held over each original 0.5 s sample. It isolates integration error.
- Thermal-only feedback refinement retains electrical, fan-lag, policy, fallback
  and history clocks at 0.5 s, and P90 updates every 5 s. This is not convergence
  of the entire coupled solver. The refined implementation is generated from the
  original online plant, with its source hash recorded.
- Parameter ranges are assumed stress tests, not measured tolerances. Fallback
  shares the perturbed parameters; plant/controller mismatch is not tested.
- Conductance ×0.8 produced three overtemperature cases, including one terminated
  at 689.5 s. Do not compare its truncated mission energy with full missions as if
  durations were equal. Failure traces are retained.
- Alarm selection uses 10 new development scenarios (4401–4410), followed by 20
  disjoint holdout scenarios (4501–4520), each with three frozen policies. The
  selected alarm targets development mean fan energy approximately; actual test
  energy differences are reported. No holdout-driven reselection is performed.
- Bootstrap resampling units are the 20 scenario means after averaging the three
  policy seeds. Windows, time steps and policy seeds are not independent trials.
- The rating illustration is one historical scenario under explicit scaling
  assumptions. It does not validate a physical 1.7 kW stack or prove optimality.
- The original source-model 750 J/K stack capacity is an assumption; no new
  parameter fitting or experimental calibration is performed here.

The staged `results/` directory contains the frozen CSV outputs and thermal
parameter snapshots. Exact file integrity is recorded in the repository-level
`SHA256SUMS.json`. Figure scripts generate SVG, PDF, PNG and 600 dpi TIFF files.
