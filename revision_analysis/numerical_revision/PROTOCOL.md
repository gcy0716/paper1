# Additional revision analysis — fixed before results, 2026-11-04

These are additional exploratory simulation analyses. Original locked results and
trained weights are retained. They do not provide experimental validation.

1. Thermal integration verification: export 20 safe-PI trajectories from the
   current locked-controller plant using ablation seeds 4301–4320. Replay its
   two-node thermal equations with recorded heat generation and fan state held
   constant within each 0.5 s interval. Compare Euler steps 0.5, 0.25, 0.1, 0.05 s
   against the exact linear two-node solution under those held inputs. First
   require 0.5 s replay to reproduce stored thermal states. This isolates thermal
   integration error; it is not a convergence test of the full feedback system.
   Before inspecting replay results, add a complementary closed-loop thermal-only
   substepping check on the same 20 safe-PI cases: 0.5/0.25/0.1/0.05 s thermal
   updates with electrical, policy, fallback and history clocks held at 0.5 s.
   Require substeps=1 to reproduce original metrics. This tests thermal feedback
   sensitivity while retaining the fixed electrical and controller discretization.
2. Parameter sensitivity: same 20 scenarios and safe-PI/fallback implementation.
   One-factor-at-a-time stack heat capacity, cathode-side capacity and both forced
   conductances at 0.8 and 1.2 times nominal; cathode capacity additionally 0.5 and
   1.5; minimum command 0.10 and 0.20 versus 0.15. These are transparent stress
   ranges, not confidence intervals or measured tolerances. Report all cases,
   including failure and 70 C termination. Fallback shares perturbed parameters,
   so this does not evaluate plant/controller parameter mismatch.
3. Rating rationale: reproduce 1.6/1.7/1.8 kW in the historical v05 worst case
   using the current online plant. Historical screening used a different stepping
   implementation; retain and distinguish both records. No universal optimum.
4. Fair alarm comparison: frozen SAC seeds 11/22/33. Development scenarios
   4401–4410, disjoint holdout 4501–4520, using the same ablation generator.
   Screen current-temperature alarm thresholds 61 to 64.5 C in 0.5 C increments.
   Select one common threshold minimizing the absolute difference in mean fan Wh
   from calibrated P90 on development only (ties: smaller threshold). Freeze it
   before testing. Report actual holdout fan-energy mismatch rather than claiming
   exact equality. Holdout contrasts: calibrated P90, selected alarm, 63 C alarm,
   default fallback. Bootstrap paired differences at scenario level (seed-averaged),
   10000 draws with seed 20261104. No interpolation between controller trajectories.
   Amendment before any development result was available: the thermal-only
   refinement changed some safe-PI peaks by up to 0.2383 C. Therefore repeat
   calibrated P90 and the development-selected alarm on the untouched holdout
   with 0.05 s thermal substeps, retaining 0.5 s controller/electrical/history
   updates. This checks a concrete numerical-sensitivity concern, without
   reselecting the alarm threshold or retraining models.
5. Integrate verified results into supplement, manuscript and responses. Preserve
   literature reconstruction, missing physical validation and missing 3D position
   evidence boundaries. No external messages or fabricated flight trajectories.
