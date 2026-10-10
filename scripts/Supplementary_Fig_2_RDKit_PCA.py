#!/usr/bin/env python3
"""Supplementary Fig. 2 — PCA of 40 chemicals in physicochemical space.

Two-panel comparison:
  (a) 10 original descriptors used in the paper's main multimodal model
      (MW, LogP, TPSA, HBD, HBA, QED, BBB_penetrant, BBB_score,
       Brenk_alerts, PAINS_alerts).
  (b) 168-descriptor extended RDKit panel (standardized, same chemicals).

With these inputs (StandardScaler + sklearn PCA) panel (a) explains 51.7% / 22.2% and
panel (b) 25.9% / 11.0% of the variance (PC1 / PC2).

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


STYLE = {  # main-text Fig. 2a (iDEP-style) design
    'Cat1':    dict(color='#A8479F', marker='o', name='Category1'),
    'Cat0':    dict(color='#E8913A', marker='s', name='Category0'),
    'Unknown': dict(color='#2BA9AA', marker='^', name='Unknown'),
}

def draw_pca(ax, X, cats, title, label):
    from scipy.spatial import ConvexHull
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    pc = pca.fit_transform(Xs)
    var = pca.explained_variance_ratio_ * 100
    handles = []
    for g in ['Cat1', 'Cat0', 'Unknown']:
        idx = [i for i, c in enumerate(cats) if c == g]
        st = STYLE[g]; col = st['color']
        xs, ys = pc[idx, 0], pc[idx, 1]
        if len(idx) >= 3:
            vals, vecs = np.linalg.eigh(np.cov(xs, ys))
            o = vals.argsort()[::-1]; vals, vecs = vals[o], vecs[:, o]
            theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
            w, h = 2 * np.sqrt(5.991 * vals)          # 95 % normal ellipse
            ax.add_patch(Ellipse((xs.mean(), ys.mean()), w, h, angle=theta,
                                 facecolor=col, alpha=0.22, edgecolor=col, lw=0.6, zorder=1))
            hull = ConvexHull(np.c_[xs, ys])
            ax.add_patch(Polygon(np.c_[xs, ys][hull.vertices], closed=True,
                                 facecolor=col, alpha=0.10, edgecolor=col, lw=0.8,
                                 ls=(0, (2, 2)), zorder=2))
        ax.scatter(xs, ys, marker=st['marker'], s=28, c=col, edgecolor='none', zorder=3)
        ax.text(xs.mean(), ys.mean(), st['name'], color=col, fontsize=9, ha='center',
                va='center', alpha=0.9, zorder=4)
        handles.append(plt.Line2D([], [], marker=st['marker'], ls='', color=col, ms=5,
                                  label=f"{st['name']} (n={len(idx)})"))
    ax.autoscale_view()
    ax.axhline(0, color='#bbbbbb', lw=0.5, zorder=0); ax.axvline(0, color='#bbbbbb', lw=0.5, zorder=0)
    ax.spines['top'].set_visible(True); ax.spines['right'].set_visible(True)
    for sp in ax.spines.values():
        sp.set_color('#c9a9c9'); sp.set_linewidth(0.8)
    ax.tick_params(top=True, right=True, labelsize=8, colors='#444444', length=2.5)
    ax.set_xlabel(f'PC1 ({var[0]:.1f}%)', fontsize=9, fontweight='normal', color='#333333')
    ax.set_ylabel(f'PC2 ({var[1]:.1f}%)', fontsize=9, fontweight='normal', color='#333333')
    ax.set_title(f'PCA: {title}', fontsize=11, fontweight='normal', loc='left', pad=30, color='#333333')
    ax.text(0, 1.045, 'Standardized descriptors; 40 chemicals', transform=ax.transAxes,
            fontsize=6.5, color='#444444', va='bottom')
    ax.text(-0.14, 1.12, label, transform=ax.transAxes, fontsize=14, fontweight='bold', va='bottom')
    ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(0, 1.0), ncol=3, fontsize=7,
              frameon=False, borderaxespad=0.1, handletextpad=0.2, columnspacing=1.0,
              bbox_transform=ax.transAxes) if False else ax.legend(handles=handles,
              loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8, frameon=False)


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
