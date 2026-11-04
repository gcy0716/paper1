# IJHE model-in-the-loop study: local reproducibility package

Manuscript HE-D-26-14868. MATLAB code was developed and run under R2023b with Reinforcement Learning Toolbox and Deep Learning Toolbox. Python analysis uses NumPy, pandas, scikit-learn and matplotlib. The scripts and frozen model files are included under `matlab_model_x650_v03/`; file hashes are in `SHA256SUMS.json`.

## What is included

- MATLAB/Python source files for the plant, scenario generators, GRU and SAC training/evaluation, statistical summaries, and figure data processing.
- Frozen seed-11/22/33 1.7 kW SAC agents, the original 10/30/60 s full-input GRUs, and six later full/minimal 10 s ablation GRUs.
- Calibration coefficients, selected scenario manifests, per-episode controller and ablation records, and stage-wise training-reward histories.
- A corrected forecast-corpus generator under `corrected_corpus_generator/`. It resolves the legacy DC-bus allocation inconsistency identified during revision.

## Important version boundary

The submitted forecast corpus and frozen GRU weights were generated with the original `run_x650_v03.m`. In 7,676 in-domain records that script calculated battery allocation before electrochemical PEMFC power clipping. The source is retained so that the submitted analyses remain auditable. `corrected_corpus_generator/run_x650_r2.m` implements allocation after clipping. The corrected 84-mission corpus is now under `corrected_probability_dataset_r2/`; the trajectory CSV is gzip compressed for transfer. Stack-temperature targets and fan histories were unchanged. The original input-only sensitivity checks remain in `revision_analysis/`. In addition, the original seed-20260910 full-input 10 s GRU was retrained on the corrected corpus with the same architecture and optimizer. Its model, calibration, training history and metrics are under `corrected_probability_dataset_r2/quantile_gru_results/`. These are post-hoc checks and do not replace the original locked analysis. Closed-loop controller evaluation used `create_pemfc_rl_env_v03.m`, which already allocates DC-bus power after electrochemical clipping.

## Reproduce the principal controller comparison

From MATLAB R2023b, set the current directory to `matlab_model_x650_v03`, then run:

```matlab
addpath(pwd)
evaluate_quantile_ablation_v01
evaluate_safety_layer_ablation_v01
evaluate_reviewer_alarm_ablation_v01
```

The first two use dedicated scenario seeds 4101–4120 and 4201–4220. The added reviewer comparison uses 4301–4320. Frozen agent MAT files are loaded by each evaluator. The original locked final-test records are supplied in `results/predictive_safety_v02`; consult the source evaluation functions and manifest before rerunning or modifying any threshold.

## Boundaries

All temperature, power and hydrogen outcomes are simulated. No measured thermal transient, cathode airflow or flight trajectory is included. The 1.7 kW architecture and several thermal parameters are conceptual. P90 is a model prediction used as a cooling trigger; its 80% two-sided interval calibration does not imply a 90% safety guarantee.

No software license has been assigned. Please contact the authors about reuse beyond inspection and reproduction of the reported analyses.

## Post-hoc corrected-predictor checks

To reconstruct the corrected training input, decompress `corrected_probability_dataset_r2/thermal_trajectories_v01.csv.gz` to a CSV with the same basename. In MATLAB R2023b, add `matlab_model_x650_v03` to the path, then run `train_quantile_gru_v01(30,128,false,fullfile(pwd,'corrected_probability_dataset_r2'),10)` and `calibrate_quantile_gru_v01(fullfile(pwd,'corrected_probability_dataset_r2'),10)`. The supplied MAT and calibration CSV are the trained artifacts used in the retained controller results; retraining can vary with the training environment.

From the repository root, add `revision_analysis/numerical_revision` to the MATLAB path. `run_numerical_revision('corrected_predictor_locked')`, `run_numerical_revision('corrected_predictor_alarm_holdout')`, `run_numerical_revision('held_predictor_alarm_holdout')` and `run_numerical_revision('held_predictor_alarm_holdout_refined')` regenerate the paired episode records in that folder's `results/`. The held variant is exploratory and keeps a triggered full-fan command for ten 0.5 s steps; it does not replace the original unheld P90 implementation. The refined variant changes only the thermal integration step to 0.05 s.

## Additional numerical revision checks

See [the numerical revision README](revision_analysis/numerical_revision/README.md) for thermal-only integration checks, one-factor sensitivity including retained failures, the illustrative rating comparison, and development-selected alarm testing on a disjoint holdout. The new analyses do not provide independent experimental validation. Python additionally requires SciPy.
