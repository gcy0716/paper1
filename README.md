# IJHE model-in-the-loop study: local reproducibility package

Manuscript HE-D-26-14868. MATLAB code was developed and run under R2023b with Reinforcement Learning Toolbox and Deep Learning Toolbox. Python analysis uses NumPy, pandas, scikit-learn and matplotlib. The scripts and frozen model files are included under `matlab_model_x650_v03/`; file hashes are in `SHA256SUMS.json`.

## What is included

- MATLAB/Python source files for the plant, scenario generators, GRU and SAC training/evaluation, statistical summaries, and figure data processing.
- Frozen seed-11/22/33 1.7 kW SAC agents, the original 10/30/60 s full-input GRUs, and six later full/minimal 10 s ablation GRUs.
- Calibration coefficients, selected scenario manifests, per-episode controller and ablation records, and stage-wise training-reward histories.
- A corrected forecast-corpus generator under `corrected_corpus_generator/`. It resolves the legacy DC-bus allocation inconsistency identified during revision.

## Important version boundary

The submitted forecast corpus and frozen GRU weights were generated with the original `run_x650_v03.m`. In 7,676 in-domain records that script calculated battery allocation before electrochemical PEMFC power clipping. The source is retained so that the submitted analyses remain auditable. `corrected_corpus_generator/run_x650_r2.m` implements allocation after clipping. The corrected 84-mission corpus and 24-mission holdout remain in the local revision workspace; their scenario manifest and compact affected-row/frozen-model audits are in `revision_analysis/`. Stack-temperature targets and fan histories were unchanged. Across the three later full-input seeds, mean in-domain MAE changed from 0.0948 to 0.0930 °C in development and 0.1160 to 0.1143 °C in the prospective holdout when the same weights received corrected inputs. The OOD and minimal-input results were unchanged; one 63 °C alarm crossing changed in each in-domain cohort, neither for an actual ≥63 °C target. The frozen models have **not** been retrained on corrected histories, and these offline checks are not new closed-loop tests. Closed-loop controller evaluation used `create_pemfc_rl_env_v03.m`, which already allocates DC-bus power after electrochemical clipping.

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

## Additional numerical revision checks

See [the numerical revision README](revision_analysis/numerical_revision/README.md) for thermal-only integration checks, one-factor sensitivity including retained failures, the illustrative rating comparison, and development-selected alarm testing on a disjoint holdout. The new analyses do not provide independent experimental validation. Python additionally requires SciPy.
