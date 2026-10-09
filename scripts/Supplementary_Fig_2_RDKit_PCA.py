#!/usr/bin/env python3
"""Supplementary Fig. 2 — PCA of 40 chemicals in physicochemical space.

Two-panel comparison:
  (a) 10 original descriptors used in the paper's main multimodal model
      (MW, LogP, TPSA, HBD, HBA, QED, BBB_penetrant, BBB_score,
       Brenk_alerts, PAINS_alerts).
  (b) 168-descriptor extended RDKit panel (standardized, same chemicals).

With these inputs (StandardScaler + sklearn PCA) panel (a) explains 51.7% / 22.2% and
panel (b) 27.0% / 11.0% of the variance (PC1 / PC2).

Run from the repository root.
Inputs
------
- data/Supplementary_Fig_2_10_descriptors_40chemicals.csv  (40 x 11: compound, MW, LogP,
    TPSA, HBD, HBA, QED, BBB_penetrant, BBB_score, Brenk_alerts, PAINS_alerts; the
    descriptors computed by scripts/ml_pipeline/compute_cactus_features.py)
- data/Supplementary_Fig_2_RDKit_descriptors_40chemicals.csv  (40 x 172: No, Chemical,
    Category, SMILES, then 168 RDKit descriptors; names in Supplementary Table 5)

Output
------
- output/Supplementary_Fig_2_RDKit_PCA.png / .pdf
"""
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Ellipse, Polygon
from fig_style import apply_style, DPI, CATEGORY_COLORS
apply_style()

# ------- Table 1 category map (40 chemicals) ---------------------------------
CAT1_CHEMS = {'Acitretin','Busulfan','Carbamazepine','Cyclophosphamide','Fluconazole',
    '5-Fluorouracil','Hydroxyurea','Isotretinoin','Methotrexate','Phenytoin',
    'Thalidomide','Topiramate','Tretinoin','Valproic acid','Pomalidomide',
    'Aspirin','Cytarabine','Ibuprofen','Trimethadione','Vismodegib'}
CAT0_CHEMS = {'Ibrutinib','Pazopanib','Ribavirin','Tacrolimus','Bosentan',
    'Cisplatin','Dabrafenib','Dasatinib','Imatinib','Rotenone',
    'Acetaminophen','Acetylsalicylic acid','D-Glucitol','L-Ascorbic acid',
    'Saccharin Sodium Salt hydrate'}
def category_of(name):
    if name in CAT1_CHEMS: return 'Cat1'
    if name in CAT0_CHEMS: return 'Cat0'
    return 'Unknown'

# The paper's 10-descriptor set (authoritative; from Molecular_Descriptors sheet)
TEN = ['MW','LogP','TPSA','HBD','HBA','QED',
       'BBB_penetrant','BBB_score','Brenk_alerts','PAINS_alerts']

CAT_MARKER = {'Cat1': ('o', 55), 'Cat0': ('s', 55), 'Unknown': ('^', 70)}


def draw_pca(ax, X, cats, title, label):
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    pc = pca.fit_transform(Xs)
    var = pca.explained_variance_ratio_ * 100

    for g in ['Cat1', 'Cat0', 'Unknown']:
        idx = [i for i, c in enumerate(cats) if c == g]
        if not idx:
            continue
        m, s = CAT_MARKER[g]
        ax.scatter(pc[idx, 0], pc[idx, 1], marker=m, s=s,
                   c=CATEGORY_COLORS[g], edgecolor='black', lw=0.5, zorder=3,
                   label=f'{g} (n={len(idx)})')
        xs, ys = pc[idx, 0], pc[idx, 1]
        if len(idx) >= 3:
            cov = np.cov(xs, ys)
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]; vals, vecs = vals[order], vecs[:, order]
            theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            w, h = 2 * np.sqrt(vals)
            ax.add_patch(Ellipse((xs.mean(), ys.mean()), w, h, angle=theta,
                                 facecolor=CATEGORY_COLORS[g], alpha=0.18,
                                 edgecolor='none', zorder=1))
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(np.c_[xs, ys])
                poly = Polygon(np.c_[xs, ys][hull.vertices], closed=True,
                               facecolor=CATEGORY_COLORS[g], alpha=0.08,
                               edgecolor=CATEGORY_COLORS[g], lw=1.0, ls='--',
                               zorder=2)
                ax.add_patch(poly)
            except Exception:
                pass

    ax.axhline(0, color='grey', lw=0.5); ax.axvline(0, color='grey', lw=0.5)
    ax.set_xlabel(f'PC1 ({var[0]:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({var[1]:.1f}%)', fontsize=12)
    ax.set_title(f'{label}  {title}', fontsize=14, loc='left', pad=10)
    ax.legend(loc='best', fontsize=10, frameon=False)


# ----- Panel (a): 10 original descriptors --------------------------------------
md = pd.read_csv('data/Supplementary_Fig_2_10_descriptors_40chemicals.csv')
cats_a = [category_of(c) for c in md['compound']]
missing = [d for d in TEN if d not in md.columns]
assert not missing, f'10-descriptor sheet missing columns: {missing}'
X10 = md[TEN].astype(float).values
print(f'(a) 10-descriptor matrix: {X10.shape}')

# ----- Panel (b): 168 extended RDKit descriptors ----------------------------
ext = pd.read_csv('data/Supplementary_Fig_2_RDKit_descriptors_40chemicals.csv')
cats_b = [{'Category1':'Cat1','Category0':'Cat0','Unknown':'Unknown'}[c] for c in ext['Category']]
descs = [c for c in ext.columns if c not in ('No','Chemical','Category','SMILES')]
assert len(descs) == 168, f'expected 168, got {len(descs)}'
X168 = ext[descs].astype(float).values
print(f'(b) 168-descriptor matrix: {X168.shape}')

# ----- Figure ---------------------------------------------------------------
fig = plt.figure(figsize=(14, 6))
gs = gridspec.GridSpec(1, 2, wspace=0.25, left=0.07, right=0.97,
                       top=0.90, bottom=0.12)
ax_a = fig.add_subplot(gs[0]); ax_b = fig.add_subplot(gs[1])
draw_pca(ax_a, X10,  cats_a, '10 original descriptors',           'a')
draw_pca(ax_b, X168, cats_b, '168 extended RDKit descriptors',    'b')

os.makedirs('output', exist_ok=True)
for fmt in ('png', 'pdf'):
    fig.savefig(f'output/Supplementary_Fig_2_RDKit_PCA.{fmt}', dpi=DPI, bbox_inches='tight',
                facecolor='white')
    print(f'saved output/Supplementary_Fig_2_RDKit_PCA.{fmt}')
plt.close(fig)
