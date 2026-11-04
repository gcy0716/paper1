"""Paired scenario-bootstrap CIs for the two post-hoc locked checks."""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'results'
ORIG = pd.read_csv(ROOT / 'matlab_model_x650_v03' / 'results' /
                   'predictive_safety_v02' / 'unified_final_test_episodes_v02.csv')
REFINED = pd.read_csv(OUT / 'locked_refined_episodes.csv')
CORRECTED = pd.read_csv(OUT / 'corrected_predictor_locked_episodes.csv')
FEASIBLE = sorted(set(range(3101, 3121)) - {3103, 3107, 3111, 3115, 3119})
RNG = np.random.default_rng(20260911)
rows = []

for analysis, source, baseline in (
    ('thermal_refined_0p05s', REFINED, 'refined_fallback'),
    ('corrected_corpus_retrain', CORRECTED, 'original_fallback'),
):
    for seed in (11, 22, 33):
        if baseline == 'refined_fallback':
            a = source[(source.agent_seed == seed) &
                       (source.variant == 'fallback')]
            b = source[(source.agent_seed == seed) &
                       (source.variant == 'p90')]
        else:
            a = ORIG[ORIG.policy == f'sac_seed{seed}_legacy']
            b = source[source.agent_seed == seed]
        a = a[a.scenario_seed.isin(FEASIBLE)].set_index('scenario_seed').sort_index()
        b = b[b.scenario_seed.isin(FEASIBLE)].set_index('scenario_seed').sort_index()
        assert list(a.index) == list(b.index) == FEASIBLE
        for metric in ('peak_temperature_C', 'fan_energy_Wh'):
            diff = (b[metric] - a[metric]).to_numpy()
            means = RNG.choice(diff, size=(10000, 15), replace=True).mean(axis=1)
            lo, hi = np.quantile(means, (0.025, 0.975))
            rows.append((analysis, seed, metric, 15, diff.mean(), lo, hi))

df = pd.DataFrame(rows, columns=('analysis', 'agent_seed', 'metric', 'n_scenarios',
                                 'mean_delta', 'ci95_low', 'ci95_high'))
df.to_csv(OUT / 'cross_review_locked_intervals.csv', index=False)
print(df.to_string(index=False))
