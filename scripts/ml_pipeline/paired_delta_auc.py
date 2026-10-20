"""Paired per-split ΔAUC (multi-modal minus transcriptomics-based), chemical-level.
Input : results/leakage_sensitivity/chem_level_folds.csv
Output: results/leakage_sensitivity/paired_delta_auc.csv (mean, SD, t-based 95% CI over the 5 splits)
Usage : python scripts/ml_pipeline/paired_delta_auc.py   (run from repository root)
"""
import pandas as pd, numpy as np
from scipy import stats
df = pd.read_csv('results/leakage_sensitivity/chem_level_folds.csv')
mods = df['mod'].unique()
print('modalities in file:', list(mods))
mm = [m for m in mods if 'multi' in m.lower()][0]
tx = [m for m in mods if m != mm][0]
rows = []
for (gs, model), g in df.groupby(['GS', 'model']):
    a = g[g['mod'] == mm].set_index('fold')['AUC']
    b = g[g['mod'] == tx].set_index('fold')['AUC']
    d = (a - b).dropna().values
    n = len(d); m = d.mean(); sd = d.std(ddof=1); h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    rows.append(dict(GS=gs, model=model, n_splits=n, mean_dAUC=m, sd=sd, ci_low=m - h, ci_high=m + h))
out = pd.DataFrame(rows)
out.to_csv('results/leakage_sensitivity/paired_delta_auc.csv', index=False)
print(out.round(3).to_string(index=False))
