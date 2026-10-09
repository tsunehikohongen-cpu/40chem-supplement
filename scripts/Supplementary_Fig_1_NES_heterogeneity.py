#!/usr/bin/env python3
"""Supplementary Fig. 1 (3 panels, (a)(b)(c) labels).
  (a) ssGSEA 30-term heatmap (formerly Fig. 2b of the main text).
      Terms selected using the Supplementary Table 1 keyword list (41 include,
      6 exclude). Top 30 by cross-compound NES variance among the 298
      keyword-matched terms. Rows ordered by Ward/Euclidean clustering.
  (b) 40x40 compound correlation heatmap.
  (c) Per-term cross-compound NES variance histogram with keyword- and
      ontology-subset medians.

Run from the repository root.
Inputs:  data/NES_matrix_GO_BP_2023_40chemicals.csv
         data/Supplementary_Table_1_keyword_list.csv
         go-basic.obo  (Gene Ontology; for the ontology-root subset, panel c). Download once:
             curl -L -o go-basic.obo https://current.geneontology.org/ontology/go-basic.obo
Output:  output/Supplementary_Fig_1_NES_heterogeneity.png / .pdf (600 dpi)
"""
import os
import re
from collections import defaultdict, deque
import numpy as np, pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from fig_style import apply_style, DPI
apply_style()

KW_CSV   = 'data/Supplementary_Table_1_keyword_list.csv'
NES_CSV  = 'data/NES_matrix_GO_BP_2023_40chemicals.csv'
OBO      = 'go-basic.obo'
ROOT     = 'GO:0032502'
N_CAT1, N_CAT0, N_UNK = 20, 15, 5
COL_C1, COL_C0, COL_UNK = '#C0392B', '#2166AC', '#1cb0b9'
VIOLIN_WITHIN, VIOLIN_BETWEEN = '#d98b80', '#9aa0a8'
ALL_COLOR, ONTO_COLOR, KW_COLOR = '#b8bcc4', '#c0392b', '#2f6fb0'

# ---- helpers ----
def go_id(term):
    m = re.search(r'\((GO:\d+)\)', term)
    return m.group(1) if m else None

def keyword_mask(terms, kw_csv):
    kw = pd.read_csv(kw_csv)
    inc = [k.lower() for r,k in zip(kw.role, kw.keyword_or_root) if r=='include']
    exc = [k.lower() for r,k in zip(kw.role, kw.keyword_or_root) if r=='exclude']
    def ok(t):
        s = re.sub(r'\s*\(GO:\d+\)','',t).lower()
        if any(e in s for e in exc): return False
        return any(i in s for i in inc)
    return np.array([ok(t) for t in terms])

def ontology_descendants(obo_path, root):
    parents, cur = {}, None
    for line in open(obo_path, encoding='utf-8'):
        line = line.rstrip()
        if line == '[Term]':
            cur = {'id': None, 'rel': [], 'obs': False}
        elif cur is not None and line.startswith('id: GO:') and cur['id'] is None:
            cur['id'] = line[4:]
        elif cur is not None and line.startswith('is_a: GO:'):
            cur['rel'].append(line.split()[1])
        elif cur is not None and line.startswith('relationship: part_of GO:'):
            cur['rel'].append(line.split()[2])
        elif cur is not None and line.startswith('is_obsolete: true'):
            cur['obs'] = True
        elif line == '' and cur is not None and cur['id']:
            if not cur['obs']:
                parents[cur['id']] = cur['rel']
            cur = None
    children = defaultdict(list)
    for c, ps in parents.items():
        for p in ps:
            children[p].append(c)
    desc, dq = set(), deque([root])
    while dq:
        n = dq.popleft()
        for c in children.get(n, []):
            if c not in desc:
                desc.add(c); dq.append(c)
    return desc

def shorten(name, maxlen=55):
    idx = name.find(' (GO:')
    if idx > 0: name = name[:idx]
    return name[:maxlen]

# ---- load ----
nes = pd.read_csv(NES_CSV, index_col=0)
assert nes.shape[1] == 40
chem_names = list(nes.columns)
no_labels = [f'No.{i+1}  {c}' for i,c in enumerate(chem_names)]
X = nes.values
kw_mask = keyword_mask(nes.index, KW_CSV)
desc = ontology_descendants(OBO, ROOT)
onto_mask = np.array([go_id(t) in desc for t in nes.index])
print(f'terms={nes.shape[0]} | keyword={kw_mask.sum()} | ontology={onto_mask.sum()}')

# ---- panel (a): top-30 by variance among 298 keyword terms, Ward clustering ----
dev_mat = nes.loc[kw_mask].astype(float)
top30 = dev_mat.var(axis=1, skipna=True).nlargest(30).index.tolist()
mat30 = nes.loc[top30].astype(float)
Z = linkage(mat30.values, method='ward', metric='euclidean')
order = dendrogram(Z, no_plot=True)['leaves']
mat_a = mat30.iloc[order]
mat_a.index = [shorten(t) for t in mat_a.index]
vabs = np.percentile(np.abs(mat_a.values), 95) or 1.0

# ---- panel (b): 40x40 compound correlation ----
corr40 = np.corrcoef(X.T)

# ---- panel (c): per-term variance histogram ----
var = X.var(axis=1)
med_all = np.median(var)
med_kw  = np.median(var[kw_mask])
med_on  = np.median(var[onto_mask])
print(f'variance medians: all={med_all:.5f} kw={med_kw:.5f} onto={med_on:.5f}')

# =========================== figure ===========================
fig = plt.figure(figsize=(16, 20))
gs = gridspec.GridSpec(2, 2, height_ratios=[1.55, 1.0], width_ratios=[1.0, 1.0],
                       hspace=0.60, wspace=0.32, left=0.09, right=0.96,
                       top=0.96, bottom=0.055)

# ---- (a) heatmap, spans top row ----
gs_a = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, :],
                                        width_ratios=[40, 0.5], wspace=0.025)
ax_a = fig.add_subplot(gs_a[0, 0])
cax_a = fig.add_subplot(gs_a[0, 1])
im = ax_a.imshow(mat_a.values, cmap='RdBu_r', vmin=-vabs, vmax=vabs,
                 aspect='auto', interpolation='nearest')
ax_a.set_xticks(range(len(chem_names)))
ax_a.set_xticklabels(no_labels, rotation=55, ha='right',
                     fontsize=10, fontweight='bold')
ax_a.set_yticks(range(len(mat_a)))
ax_a.set_yticklabels(mat_a.index, fontsize=10.5, fontweight='bold')
ax_a.set_xlabel('Chemical (No.1-40; Cat1 | Cat0 | Unknown)',
                fontweight='bold', fontsize=12)
ax_a.set_ylabel('GO BP 2023 term', fontweight='bold', fontsize=12, labelpad=6)
# category dividers
for b in (N_CAT1 - 0.5, N_CAT1 + N_CAT0 - 0.5):
    ax_a.axvline(b, color='black', lw=1.3, ls='--')
ax_a.set_xticks(np.arange(-0.5, len(chem_names), 1), minor=True)
ax_a.set_yticks(np.arange(-0.5, len(mat_a), 1), minor=True)
ax_a.grid(which='minor', color='white', linewidth=0.4)
ax_a.tick_params(which='minor', bottom=False, left=False)
# category labels above heatmap
ax_a.text(N_CAT1/2 - 0.5, -0.9, 'Category 1', ha='center', va='bottom',
          fontsize=12, fontweight='bold', color=COL_C1)
ax_a.text(N_CAT1 + N_CAT0/2 - 0.5, -0.9, 'Category 0', ha='center', va='bottom',
          fontsize=12, fontweight='bold', color=COL_C0)
ax_a.text(N_CAT1 + N_CAT0 + N_UNK/2 - 0.5, -0.9, 'Unknown', ha='center', va='bottom',
          fontsize=12, fontweight='bold', color=COL_UNK)
cb = fig.colorbar(im, cax=cax_a)
cb.set_label('NES', fontweight='bold', fontsize=11)
cb.ax.tick_params(labelsize=9)
# align 'a' label with the natural x position of the y-axis label (left of row labels)
fig.canvas.draw()
ylabel_disp = ax_a.yaxis.label.get_window_extent()
ylabel_x_axes = ax_a.transAxes.inverted().transform((ylabel_disp.x0 + ylabel_disp.width/2, 0))[0]
ax_a.text(ylabel_x_axes, 1.03, 'a', transform=ax_a.transAxes,
          fontsize=22, fontweight='bold', va='top', ha='center')

# ---- (b) 40x40 correlation ----
ax_b = fig.add_subplot(gs[1, 0])
im_b = ax_b.imshow(corr40, cmap='RdBu_r', vmin=-0.5, vmax=0.5, aspect='equal')
for b in (N_CAT1, N_CAT1 + N_CAT0):
    ax_b.axhline(b - 0.5, color='black', lw=1.2)
    ax_b.axvline(b - 0.5, color='black', lw=1.2)
ax_b.set_xticks([]); ax_b.set_yticks([])
ax_b.text(N_CAT1/2 - 0.5, -1.8, 'Cat1', ha='center', fontsize=12, fontweight='bold', color=COL_C1)
ax_b.text(N_CAT1 + N_CAT0/2 - 0.5, -1.8, 'Cat0', ha='center', fontsize=12, fontweight='bold', color=COL_C0)
ax_b.text(N_CAT1 + N_CAT0 + N_UNK/2 - 0.5, -1.8, 'Unknown', ha='center', fontsize=12, fontweight='bold', color=COL_UNK)
ax_b.set_title('Compound × compound NES correlation', fontsize=12, pad=24, fontweight='bold')
cb_b = fig.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04)
cb_b.set_label('r', fontweight='bold')
cb_b.ax.tick_params(labelsize=10)
ax_b.text(-0.10, 1.08, 'b', transform=ax_b.transAxes,
          fontsize=22, fontweight='bold', va='top', ha='left')

# ---- (c) variance histogram ----
ax_c = fig.add_subplot(gs[1, 1])
ax_c.hist(var, bins=np.linspace(0, 0.05, 45), color=ALL_COLOR, edgecolor='none')
ax_c.axvline(med_all, ls='--', color='#555', lw=1.2)
ax_c.axvline(med_on,  color=ONTO_COLOR, lw=1.6)
ax_c.axvline(med_kw,  color=KW_COLOR,   lw=1.6)
ax_c.legend(handles=[
    Patch(facecolor=ALL_COLOR, label=f'All terms (n={nes.shape[0]})'),
    plt.Line2D([],[], color=ONTO_COLOR, label=f'Ontology median (n={int(onto_mask.sum())})'),
    plt.Line2D([],[], color=KW_COLOR,   label=f'Keyword median (n={int(kw_mask.sum())})'),
], fontsize=10, frameon=False, loc='upper right')
ax_c.set_xlim(0, 0.05)
ax_c.set_xlabel('Cross-compound variance of NES (per term)', fontweight='bold', fontsize=11)
ax_c.set_ylabel('Term count', fontweight='bold', fontsize=11)
ax_c.tick_params(labelsize=10)
ax_c.set_title('Developmental terms are not more variable\nthan the genome-wide term pool',
               fontsize=12, pad=12, fontweight='bold')
for s in ('top','right'): ax_c.spines[s].set_visible(False)
ax_c.text(-0.14, 1.08, 'c', transform=ax_c.transAxes,
          fontsize=22, fontweight='bold', va='top', ha='left')

os.makedirs('output', exist_ok=True)
for ext in ('png','pdf'):
    path = f'output/Supplementary_Fig_1_NES_heterogeneity.{ext}'
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    print('saved:', path)
plt.close(fig)
