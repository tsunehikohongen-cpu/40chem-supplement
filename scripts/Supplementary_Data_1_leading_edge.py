#!/usr/bin/env python3
"""Supplementary Data 1 — Gene-level attribution of the ssGSEA NES of the 30
developmental GO BP terms (shown as the heatmap in Supplementary Fig. 1a;
formerly Fig. 2b of the main text).

For every chemical × developmental GO BP term of that heatmap, this script
emits the leading-edge genes with per-gene rank and contribution.

Run from the repository root:  python scripts/Supplementary_Data_1_leading_edge.py

Pipeline
--------
1. Load `40 Chem DESeq_normalized.csv` (gene × 126 sample log2-DESeq matrix;
   columns: 40 chemicals × 3 doses + 6 DMSO vehicles).
2. For each chemical, pick its highest-dose column; compute
   Δrlog = chemical − mean(DMSO 0.1%).
3. Run `gseapy.ssgsea` against Enrichr `GO_Biological_Process_2023` to get
   the full 2,692 term × 40 chemical NES matrix (also used by Fig 5, Fig 6,
   Supp Fig 1). *This step is optional here — the NES matrix is shipped as
   `data/NES_matrix_GO_BP_2023_40chemicals.csv`.*
4. Pick the 30 developmental GO BP terms of Supplementary Fig. 1a (keyword-filter
   defined in Supplementary Table 1, then top-30 by cross-compound NES
   variance).
5. For every chemical × term, re-walk the delta-ranked gene list, compute the
   ssGSEA running sum (hit weight = |Δrlog|^α, α=0.25), find the peak and
   extract leading-edge genes (positive-ES: genes in set at or before peak;
   negative-ES: at or after peak).

Output
------
- output/Supplementary_Data_1_leading_edge.xlsx  (3 sheets: README, LeadingEdge_summary, per_gene)
- output/Supplementary_Data_1_leading_edge_per_gene.csv  (the per-gene sheet as a standalone CSV)
  (the committed copies of these two files are in data/)

Inputs required
---------------
- data/40_Chem_DESeq_normalized.csv  (StrandNGS DESeq-normalized log2 matrix)
- data/NES_matrix_GO_BP_2023_40chemicals.csv  (ssGSEA NES matrix; used only to pick the 30 terms)
- data/Supplementary_Table_1_keyword_list.csv  (Supplementary Table 1)
"""
import os
import re
import numpy as np
import pandas as pd
import gseapy as gp

ALPHA = 0.25
DMSO_PATTERN = r'^DMSO.*\s0\.1$'   # highest DMSO dose

HIGHDOSE = {
    'Acitretin':'Acitretin_1','Busulfan':'Busulfan_1','Carbamazepine':'Carbamazepine_1',
    'Cyclophosphamide':'Cyclophosphamide_1','Fluconazole':'Fluconazole_1',
    '5-Fluorouracil':'5-Fluorouracil_1','Hydroxyurea':'Hydroxyurea_1',
    'Isotretinoin':'Isotretinoin_1','Methotrexate':'Methotrexate_1',
    'Phenytoin':'Phenytoin_1','Thalidomide':'Thalidomide_1','Topiramate':'Topiramate_1',
    'Tretinoin':'Tretinoin_1','Valproic acid':'Valproic acid_1',
    'Pomalidomide':'Pomalidomide_1','Aspirin':'Aspirin_1',
    'Cytarabine':'Cytarabine_0.3',                 # top dose for this one is 0.3
    'Ibuprofen':'Ibuprofen_1','Trimethadione':'Trimethadione_1','Vismodegib':'Vismodegib_1',
    'Ibrutinib':'Ibrutinib_1','Pazopanib':'Pazopanib_1','Ribavirin':'Ribavirin_1',
    'Tacrolimus':'Tacrolimus_1','Bosentan':'Bosentan_1','Cisplatin':'Cisplatin_1',
    'Dabrafenib':'Dabrafenib 1','Dasatinib':'Dasatinib 0.1','Imatinib':'Imatinib 1',
    'Acetylsalicylic acid':'Acetylsalicylic acid 1','D-Glucitol':'D-Glucitol 1',
    'L-Ascorbic acid':'L-Ascorbic acid 1',
    'Saccharin Sodium Salt hydrate':'Saccharin Sodium Salt hydrate 1',
    'Deltamethrin':'Deltamethrin 1',
    "2,2',4,4',5-Pentabromodiphenyl ether": "2,2',4,4',5-Pentabromodiphenyl ether 1",
    'Bisphenol A':'Bisphenol A 1','Chlorpyrifos':'Chlorpyrifos 1',
    'imidacloprid':'imidacloprid 1','Acetaminophen':'Acetaminophen 1',
    'Rotenone':'Rotenone 25',
}


def load_delta(deseq_csv):
    df = pd.read_csv(deseq_csv, index_col=0)
    if df.index.duplicated().any():
        df = df.groupby(df.index).max()                      # collapse by max
    dmso_cols = [c for c in df.columns if re.match(DMSO_PATTERN, c)]
    dmso_mean = df[dmso_cols].mean(axis=1)
    delta = pd.DataFrame({n: df[col] - dmso_mean for n, col in HIGHDOSE.items()})
    return delta


def pick_fig2b_terms(nes_csv, keyword_csv, n=30):
    nes = pd.read_csv(nes_csv, index_col=0)
    kw = pd.read_csv(keyword_csv)
    inc = [k.lower() for r, k in zip(kw.role, kw.keyword_or_root) if r == 'include']
    exc = [k.lower() for r, k in zip(kw.role, kw.keyword_or_root) if r == 'exclude']
    def ok(t):
        s = re.sub(r'\s*\(GO:\d+\)', '', t).lower()
        if any(e in s for e in exc): return False
        return any(i in s for i in inc)
    kw_terms = nes.index[[ok(t) for t in nes.index]]
    return nes.loc[kw_terms].var(axis=1).nlargest(n).index.tolist()


def leading_edge(delta, terms, gene_sets):
    rows_sum, rows_pg = [], []
    for term in terms:
        gs = set(gene_sets.get(term, []))
        if not gs:
            continue
        for chem in delta.columns:
            s = delta[chem].sort_values(ascending=False)
            genes = s.index.values; vals = s.values; N = len(genes)
            in_set = np.array([g in gs for g in genes])
            size = int(in_set.sum())
            if size == 0: continue
            hw = np.abs(vals) ** ALPHA
            hit_sum = hw[in_set].sum()
            miss = 1.0 / (N - size)
            run = np.empty(N); cur = 0.0
            for i in range(N):
                cur += (hw[i] / hit_sum) if in_set[i] else -miss
                run[i] = cur
            peak_i = int(np.argmax(np.abs(run))); peak_val = float(run[peak_i])
            lead = in_set.copy()
            if peak_val >= 0:
                lead[peak_i + 1:] = False
            else:
                lead[:peak_i] = False
            idx = np.where(lead)[0]
            rows_sum.append({
                'chemical': chem, 'term': term,
                'gene_set_size': size, 'n_leading_edge': int(lead.sum()),
                'ES_at_peak': peak_val, 'peak_rank': peak_i + 1,
                'leading_edge_genes': ';'.join(genes[idx][:50]),
            })
            contrib = hw / hit_sum
            for i in idx:
                rows_pg.append({
                    'chemical': chem, 'term': term, 'gene': genes[i],
                    'rank': int(i + 1), 'delta_rlog': float(vals[i]),
                    'contribution': float(contrib[i]),
                })
    return pd.DataFrame(rows_sum), pd.DataFrame(rows_pg)


if __name__ == '__main__':
    delta = load_delta('data/40_Chem_DESeq_normalized.csv')
    print(f'delta: {delta.shape}')
    terms30 = pick_fig2b_terms('data/NES_matrix_GO_BP_2023_40chemicals.csv',
                               'data/Supplementary_Table_1_keyword_list.csv')
    print(f'developmental terms (Supplementary Fig. 1a): {len(terms30)}')
    gs = gp.get_library('GO_Biological_Process_2023', organism='Human')
    summary, per_gene = leading_edge(delta, terms30, gs)
    print(f'summary: {len(summary)}  per-gene: {len(per_gene)}')

    readme = pd.DataFrame({
        'field': ['Title', 'Description', 'Terms', 'Chemicals',
                  'ssGSEA input', 'ssGSEA library', 'Weight alpha',
                  'Leading-edge definition', 'Source code'],
        'value': [
            'Supplementary Data 1 — Gene-level attribution of the ssGSEA NES of the 30 developmental GO BP terms (Supplementary Fig. 1a; formerly Fig. 2b)',
            'For every chemical × developmental GO BP term of the heatmap, '
            'this file lists the leading-edge genes with per-gene rank and contribution.',
            '30 (keyword-filtered; top-30 by cross-compound NES variance)',
            '40 (highest-dose per chemical)',
            'Δrlog = chemical (highest dose) − mean(DMSO 0.1%)',
            'Enrichr GO_Biological_Process_2023',
            '0.25',
            'Positive-ES: genes in set at or before running-sum peak. '
            'Negative-ES: genes in set at or after peak.',
            'Supplementary_Data_1_leading_edge.py',
        ]
    })
    os.makedirs('output', exist_ok=True)
    with pd.ExcelWriter('output/Supplementary_Data_1_leading_edge.xlsx', engine='openpyxl') as w:
        readme.to_excel(w, sheet_name='README', index=False)
        summary.to_excel(w, sheet_name='LeadingEdge_summary', index=False)
        per_gene.to_excel(w, sheet_name='per_gene', index=False)
    per_gene.to_csv('output/Supplementary_Data_1_leading_edge_per_gene.csv', index=False)
    print('saved: output/Supplementary_Data_1_leading_edge.xlsx + output/Supplementary_Data_1_leading_edge_per_gene.csv')
