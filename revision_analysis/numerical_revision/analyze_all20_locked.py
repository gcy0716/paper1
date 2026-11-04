"""Unscreened and conditionally screened locked-controller contrasts."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'matlab_model_x650_v03' / 'results' / 'predictive_safety_v02' / 'unified_final_test_episodes_v02.csv'
OUT = Path(__file__).resolve().parent / 'results' / 'all20_locked_contrasts.csv'
data = pd.read_csv(SOURCE)
rows = []
for scope, ids in (
    ('all20', list(range(3101, 3121))),
    ('screened15', sorted(set(range(3101, 3121)) - {3103, 3107, 3111, 3115, 3119})),
):
    rng = np.random.default_rng(20260911)
    for seed in (11, 22, 33):
        for name, left, right in (
            ('P90 - fallback', f'sac_seed{seed}_legacy', f'sac_seed{seed}_p90'),
            ('P90-SAC - MPC', 'mpc', f'sac_seed{seed}_p90'),
        ):
            a = data[data.policy == left].set_index('scenario_seed').loc[ids]
            b = data[data.policy == right].set_index('scenario_seed').loc[ids]
            for metric in ('peak_temperature_C', 'fan_energy_Wh'):
                delta = (b[metric] - a[metric]).to_numpy()
                draws = rng.choice(delta, size=(10000, len(ids)), replace=True).mean(axis=1)
                lo, hi = np.quantile(draws, (0.025, 0.975))
                rows.append((scope, seed, name, metric, len(ids), delta.mean(), lo, hi))
result = pd.DataFrame(rows, columns=('scope', 'agent_seed', 'contrast', 'metric',
                                     'n_scenarios', 'mean_delta', 'ci95_low', 'ci95_high'))
result.to_csv(OUT, index=False)
print(result.to_string(index=False))
