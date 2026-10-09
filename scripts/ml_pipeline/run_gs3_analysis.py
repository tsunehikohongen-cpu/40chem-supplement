"""
GeneSet3 (Top190 genes) ML Analysis
Models: Logistic Regression, SVM, Random Forest
Modalities: Transcriptomics-based (Non-CACTUS) / Multi-modal (CACTUS, +10 descriptors)
CV: StratifiedShuffleSplit, n_splits=5, test_size=0.5, compound-level

Output per modality:
  - CV_metrics_{modality}_GeneSet190.csv   : AUC/Accuracy/F1/MCC/TP/TN/FP/FN per fold + Mean±SD
  - CV_folds_{modality}_GeneSet190/fold{1-5}/input_data{1,2}.csv
  - CV_folds_{modality}_GeneSet190/fold{1-5}/output_data{1,2}.csv
  - FinalPred_{modality}_GeneSet190/input_data{1,2}.csv + output_data1.csv
"""

# ═══════════════════════════════════════════════════════════════════
#  CONFIG — edit these paths before running
# ═══════════════════════════════════════════════════════════════════
INPUT_DIR  = './inputs'    # directory containing all input files listed below
OUTPUT_DIR = './outputs'   # all results will be written here

# Required input files (place in INPUT_DIR):
#   Top190_data.csv               — 190 genes × (compound×conc) logFC matrix;
#                                   rows=genes (column "symbol"), cols=samples
#   Table1_labels_A_B.xlsx        — compound labels (Label A: 1=positive, 0=negative)
#   cactus_features_40chem.csv    — molecular descriptors (MW, LogP, TPSA, ...)
# ═══════════════════════════════════════════════════════════════════

import os, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             f1_score, matthews_corrcoef, confusion_matrix)
warnings.filterwarnings('ignore')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Load Top190 gene expression data ──────────────────────────────────
print("Loading Top190_data.csv ...")
raw = pd.read_csv(os.path.join(INPUT_DIR, 'Top190_data.csv'))
genes_190 = [s.strip() for s in raw['symbol'].tolist()]
print(f"  {len(genes_190)} genes")

# ── 2. Compound column mapping ────────────────────────────────────────────
# Maps Top190 column prefix → (compound_no, display_name, is_labeled)
PFXMAP = {
    'Acitretin':                       (1,  'Acitretin',                         True),
    'Busulfan':                        (2,  'Busulfan',                          True),
    'Carbamazepine':                   (3,  'Carbamazepine',                     True),
    'Cyclophosphamide':                (4,  'Cyclophosphamide',                  True),
    'Fluconazole':                     (5,  'Fluconazole',                       True),
    'X5Fluorouracil':                  (6,  '5-Fluorouracil',                    True),
    'Hydroxyurea':                     (7,  'Hydroxyurea',                       True),
    'Isotretinoin':                    (8,  'Isotretinoin',                      True),
    'Methotrexate':                    (9,  'Methotrexate',                      True),
    'Phenytoin':                       (10, 'Phenytoin',                         True),
    'Thalidomide':                     (11, 'Thalidomide',                       True),
    'Topiramate':                      (12, 'Topiramate',                        True),
    'Tretinoin':                       (13, 'Tretinoin',                         True),
    'Valproicacid':                    (14, 'Valproic acid',                     True),
    'Pomalidomide':                    (15, 'Pomalidomide',                      True),
    'Aspirin':                         (16, 'Aspirin',                           True),
    'Cytarabine':                      (17, 'Cytarabine',                        True),
    'Ibuprofen':                       (18, 'Ibuprofen',                         True),
    'Trimethadione':                   (19, 'Trimethadione',                     True),
    'Vismodegib':                      (20, 'Vismodegib',                        True),
    'Ibrutinib':                       (21, 'Ibrutinib',                         True),
    'Pazopanib':                       (22, 'Pazopanib',                         True),
    'Ribavirin':                       (23, 'Ribavirin',                         True),
    'Tacrolimus':                      (24, 'Tacrolimus',                        True),
    'Bosentan':                        (25, 'Bosentan',                          True),
    'Cisplatin':                       (26, 'Cisplatin',                         True),
    'Dabrafenib':                      (27, 'Dabrafenib',                        True),
    'Dasatinib':                       (28, 'Dasatinib',                         True),
    'Imatinib':                        (29, 'Imatinib',                          True),
    'Rotenone':                        (30, 'Rotenone',                          True),
    'Acetaminophen':                   (31, 'Acetaminophen',                     True),
    'Acetylsalicylicacid':             (32, 'Acetylsalicylic acid',              True),
    'DGlucitol':                       (33, 'D-Glucitol',                        True),
    'LAscorbicacid':                   (34, 'L-Ascorbic acid',                   True),
    'SaccharinSodiumSalthydrate':      (35, 'Saccharin Sodium Salt hydrate',     True),
    'Deltamethrin':                    (36, 'Deltamethrin',                      False),
    'X22445Pentabromodiphenylether':   (37, "2,2',4,4',5-Pentabromodiphenyl ether", False),
    'BisphenolA':                      (38, 'Bisphenol A',                       False),
    'Chlorpyrifos':                    (39, 'Chlorpyrifos',                      False),
    'imidacloprid':                    (40, 'imidacloprid',                      False),
}

SUFFIX_DECIMAL = {
    '25': '2.5', '1': '1', '03': '0.3', '01': '0.1',
    '003': '0.03', '0003': '0.003', '001': '0.01',
}

# Build gene expression matrix (genes × samples)
data_cols = [c for c in raw.columns if c not in
             ['Unnamed: 0', 'User_ID', 'ensembl_ID', 'symbol',
              'DMSO101', 'DMSO1001', 'DMSO10001', 'DMSO201', 'DMSO2001', 'DMSO20001']]
expr = raw[['symbol'] + data_cols].copy()
expr['symbol'] = expr['symbol'].str.strip()
expr = expr.set_index('symbol')   # rows=genes, cols=compound×conc

def get_compound_cols(pfx):
    """Find sample columns for a compound prefix (handles both _ and no-_ separators)."""
    return [c for c in data_cols if c == pfx or
            c.startswith(pfx + '_') or
            (c.startswith(pfx) and not any(
                c.startswith(p) for p in PFXMAP if p != pfx and len(p) > len(pfx)))]

def cols_to_samples(pfx, disp, cols):
    """Return [(sample_name, col)] sorted by ascending concentration."""
    result = []
    for c in cols:
        suffix  = c[len(pfx):].lstrip('_')
        decimal = SUFFIX_DECIMAL.get(suffix, suffix)
        result.append((float(decimal), f'{disp}_{decimal}', c))
    result.sort(key=lambda x: x[0])
    return [(sn, c) for _, sn, c in result]

# Build full sample × gene matrix (all 40 compounds)
all_rows = []
for pfx in sorted(PFXMAP, key=lambda p: PFXMAP[p][0]):
    no, disp, is_labeled = PFXMAP[pfx]
    cols = get_compound_cols(pfx)
    if not cols:
        print(f"  WARNING: no columns found for {pfx}")
        continue
    for sn, c in cols_to_samples(pfx, disp, cols):
        all_rows.append({'_sample': sn, '_no': no, '_labeled': is_labeled,
                         **dict(zip(genes_190, expr[c].values))})

df_all = pd.DataFrame(all_rows).set_index('_sample')
print(f"  Full matrix: {df_all.shape}")

X_all_raw = df_all[genes_190]
meta_all  = df_all[['_no', '_labeled']]

# ── 3. Labels ─────────────────────────────────────────────────────────────
df_lbl  = pd.read_excel(os.path.join(INPUT_DIR, 'Table1_labels_A_B.xlsx'),
                        sheet_name='Labels A_B', header=None)
chem_df = df_lbl.iloc[3:43, :5].copy()
chem_df.columns = ['No', 'Chemical', 'Source', 'Label_A', 'Label_B']
chem_df['No'] = chem_df['No'].astype(int)
label_lookup = {}
for _, row in chem_df.iterrows():
    try:
        label_lookup[int(row['No'])] = int(row['Label_A'])
    except (ValueError, TypeError):
        pass

labeled_mask = meta_all['_labeled'].astype(bool)
X_labeled_raw = X_all_raw[labeled_mask]
y_labeled     = np.array([label_lookup[int(no)] for no in
                           meta_all.loc[labeled_mask, '_no']])
nos_labeled   = meta_all.loc[labeled_mask, '_no'].astype(int).tolist()
nos_unique    = sorted(set(nos_labeled))
X_unk_raw     = X_all_raw[~labeled_mask]

print(f"  Labeled: {X_labeled_raw.shape}  pos={y_labeled.sum()}  neg={(y_labeled==0).sum()}")
print(f"  Unknown: {X_unk_raw.shape}")

# ── 4. CACTUS molecular descriptors ──────────────────────────────────────
cactus_df  = pd.read_csv(os.path.join(INPUT_DIR, 'cactus_features_40chem.csv'))
CACTUS_COLS = ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'QED',
               'BBB_penetrant', 'BBB_score', 'Brenk_alerts', 'PAINS_alerts']
cactus_lookup = {r['compound'].strip().lower(): {c: r[c] for c in CACTUS_COLS}
                 for _, r in cactus_df.iterrows()}
CACTUS_NAME_MAP = {
    '5-Fluorouracil':                      '5-fluorouracil',
    'Valproic acid':                       'valproic acid',
    'Acetylsalicylic acid':                'acetylsalicylic acid',
    'D-Glucitol':                          'd-glucitol',
    'L-Ascorbic acid':                     'l-ascorbic acid',
    'Saccharin Sodium Salt hydrate':       'saccharin',
    "2,2',4,4',5-Pentabromodiphenyl ether":"2,2',4,4',5-pentabromodiphenyl ether",
    'Bisphenol A':                         'bisphenol a',
}

def get_cactus_row(sample_name):
    disp = sample_name.rsplit('_', 1)[0]
    key  = CACTUS_NAME_MAP.get(disp, disp.lower())
    return cactus_lookup.get(key, {c: np.nan for c in CACTUS_COLS})

def add_cactus(X_raw):
    cac_rows = [get_cactus_row(s) for s in X_raw.index]
    return pd.concat([X_raw, pd.DataFrame(cac_rows, index=X_raw.index)], axis=1)

X_cac_labeled_raw = add_cactus(X_labeled_raw)
X_cac_unk_raw     = add_cactus(X_unk_raw)
print(f"  CACTUS labeled: {X_cac_labeled_raw.shape}")

# ── 5. ML models ──────────────────────────────────────────────────────────
def make_models():
    return {
        'LR':  LogisticRegression(C=0.1, max_iter=2000,
                                   class_weight='balanced', random_state=42),
        'SVM': SVC(kernel='rbf', C=1.0, probability=True,
                   class_weight='balanced', random_state=42),
        'RF':  RandomForestClassifier(n_estimators=200,
                                      class_weight='balanced_subsample',
                                      random_state=42),
    }

# ── 6. Save fold files ────────────────────────────────────────────────────
def save_fold(X_tr, y_tr, X_te, y_te, fold_dir):
    os.makedirs(fold_dir, exist_ok=True)
    X_tr.to_csv(os.path.join(fold_dir, 'input_data1.csv'))
    pd.DataFrame({'Category': y_tr}).to_csv(
        os.path.join(fold_dir, 'output_data1.csv'), index=False)
    X_te.to_csv(os.path.join(fold_dir, 'input_data2.csv'))
    pd.DataFrame({'Category': y_te}).to_csv(
        os.path.join(fold_dir, 'output_data2.csv'), index=False)

# ── 7. CV ──────────────────────────────────────────────────────────────────
def run_cv(X_raw, y, nos_per_sample, fold_dir_base, tag):
    """
    Compound-level StratifiedShuffleSplit CV.
    TP/TN/FP/FN computed directly from confusion_matrix.
    """
    nos_arr  = np.array(nos_per_sample)
    uniq_nos = np.array(sorted(set(nos_per_sample)))
    chem_lbls = np.array([label_lookup[n] for n in uniq_nos])
    idx_map  = {no: np.where(nos_arr == no)[0] for no in uniq_nos}

    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=42)
    records = []

    for fi, (tr_cidx, te_cidx) in enumerate(
            sss.split(np.zeros(len(uniq_nos)), chem_lbls), start=1):
        tr_nos  = uniq_nos[tr_cidx]
        te_nos  = uniq_nos[te_cidx]
        tr_mask = np.isin(nos_arr, tr_nos)
        te_mask = np.isin(nos_arr, te_nos)

        X_tr_r, y_tr = X_raw.iloc[tr_mask], y[tr_mask]
        X_te_r, y_te = X_raw.iloc[te_mask], y[te_mask]

        save_fold(X_tr_r, y_tr, X_te_r, y_te,
                  os.path.join(fold_dir_base, f'fold{fi}'))

        sc = StandardScaler()
        Xtr_s = sc.fit_transform(X_tr_r.values)
        Xte_s = sc.transform(X_te_r.values)

        print(f"  {tag} Fold {fi}: train={len(tr_nos)}cpds ({y_tr.sum()}pos/{(y_tr==0).sum()}neg)  "
              f"test={len(te_nos)}cpds ({y_te.sum()}pos/{(y_te==0).sum()}neg)")

        for mname, clf in make_models().items():
            clf.fit(Xtr_s, y_tr)
            yp    = clf.predict(Xte_s)
            yprob = clf.predict_proba(Xte_s)[:, 1]
            try:
                auc = roc_auc_score(y_te, yprob)
            except Exception:
                auc = np.nan
            tn, fp, fn, tp = confusion_matrix(y_te, yp, labels=[0, 1]).ravel()
            records.append({
                'Fold': fi, 'Model': mname,
                'AUC': auc,
                'Accuracy': accuracy_score(y_te, yp),
                'F1':  f1_score(y_te, yp, zero_division=0),
                'MCC': matthews_corrcoef(y_te, yp),
                'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
            })

    # Summarise
    summary = []
    print(f"\n  {'Model':5s} {'Metric':10s}  F1      F2      F3      F4      F5      Mean±SD")
    for mname in ['LR', 'SVM', 'RF']:
        sub = [r for r in records if r['Model'] == mname]
        for metric in ['AUC', 'Accuracy', 'F1', 'MCC', 'TP', 'TN', 'FP', 'FN']:
            vals = [r[metric] for r in sub]
            mn   = float(np.nanmean(vals))
            sd   = float(np.nanstd(vals))
            summary.append({'Model': mname, 'Metric': metric, 'GeneSet': 'GeneSet190',
                             **{f'Fold{i+1}': vals[i] for i in range(5)},
                             'Mean': mn, 'SD': sd})
        for metric in ['AUC', 'Accuracy', 'F1', 'MCC']:
            vals = [r[metric] for r in sub]
            fs   = '  '.join(f'{v:.3f}' for v in vals)
            print(f"  {mname:5s} {metric:10s}  {fs}  {np.nanmean(vals):.3f}±{np.nanstd(vals):.3f}")

    return pd.DataFrame(summary)

# ── 8. Save final prediction inputs ──────────────────────────────────────
def save_final_pred_inputs(X_lab, y_lab, X_unk, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    X_lab.to_csv(os.path.join(out_dir, 'input_data1.csv'))
    pd.DataFrame({'Category': y_lab}).to_csv(
        os.path.join(out_dir, 'output_data1.csv'), index=False)
    X_unk.to_csv(os.path.join(out_dir, 'input_data2.csv'))
    print(f"  Final_pred inputs saved → {out_dir}")

# ══════════════════════════════════════════════════════════════════════════
# RUN ANALYSES
# ══════════════════════════════════════════════════════════════════════════

print(f"\n{'='*65}")
print("  Transcriptomics-based (Non-CACTUS) / GeneSet3 (190 genes)")
print(f"{'='*65}")
nc_cv_dir = os.path.join(OUTPUT_DIR, 'CV_folds_NonCACTUS_GeneSet190', 'CV_folds')
df_nc = run_cv(X_labeled_raw, y_labeled, nos_labeled,
               nc_cv_dir, 'NonCACTUS_GS190')
df_nc.to_csv(os.path.join(OUTPUT_DIR, 'CV_metrics_NonCACTUS_GeneSet190.csv'), index=False)
save_final_pred_inputs(X_labeled_raw, y_labeled, X_unk_raw,
                       os.path.join(OUTPUT_DIR, 'FinalPred_NonCACTUS_GeneSet190'))

print(f"\n{'='*65}")
print("  Multi-modal (CACTUS) / GeneSet3 (190+10 = 200 features)")
print(f"{'='*65}")
cac_cv_dir = os.path.join(OUTPUT_DIR, 'CV_folds_CACTUS_GeneSet190', 'CV_folds')
df_cac = run_cv(X_cac_labeled_raw, y_labeled, nos_labeled,
                cac_cv_dir, 'CACTUS_GS190')
df_cac.to_csv(os.path.join(OUTPUT_DIR, 'CV_metrics_CACTUS_GeneSet190.csv'), index=False)
save_final_pred_inputs(X_cac_labeled_raw, y_labeled, X_cac_unk_raw,
                       os.path.join(OUTPUT_DIR, 'FinalPred_CACTUS_GeneSet190'))

print("\n✓ All GeneSet3 analyses complete.")
