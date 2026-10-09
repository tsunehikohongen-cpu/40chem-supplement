"""
GeneSet1 (Top150) / GeneSet2 (Top182) ML Analysis
Models: Logistic Regression, SVM, Random Forest
Modalities: Transcriptomics-based (Non-CACTUS) / Multi-modal (CACTUS)
CV: StratifiedShuffleSplit, n_splits=5, test_size=0.5, compound-level

Analyses performed:
  1. Non-CACTUS GeneSet1 (150 genes)
  2. CACTUS      GeneSet2 (182 genes)   [re-analysis with StratifiedShuffleSplit]
  3. CACTUS      GeneSet1 (150 genes)

Output per analysis:
  - CV_metrics.csv     : AUC / Accuracy / F1 / MCC / TP / TN / FP / FN per fold + Mean±SD
  - CV_folds/fold{1-5}/input_data{1,2}.csv   : train / test feature matrices
  - CV_folds/fold{1-5}/output_data{1,2}.csv  : train / test labels
  - Final_prediction/predictions.csv          : unknown compound predictions (Non-CACTUS only)
"""

# ═══════════════════════════════════════════════════════════════════
#  CONFIG — edit these paths before running
# ═══════════════════════════════════════════════════════════════════
INPUT_DIR  = './inputs'    # directory containing all input files listed below
OUTPUT_DIR = './outputs'   # all results will be written here

# Required input files (place in INPUT_DIR):
#   40 Chem logFC-ac99b402.csv    — RNA-seq logFC matrix (genes × compound×conc)
#   Table1_labels_A_B.xlsx        — compound labels (Label A: 1=positive, 0=negative)
#   CytoHubba_Top150_GeneSet.csv  — GeneSet1 gene list (column: "name")
#   GeneSet182.csv                — GeneSet2 gene list (column: "Gene")
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

# ── 1. Load data ──────────────────────────────────────────────────────────
print("Loading logFC matrix ...")
logfc = pd.read_csv(os.path.join(INPUT_DIR, '40 Chem logFC-ac99b402.csv'), index_col=0)
print(f"  logFC: {logfc.shape}")

print("Loading compound labels ...")
df_raw  = pd.read_excel(os.path.join(INPUT_DIR, 'Table1_labels_A_B.xlsx'),
                        sheet_name='Labels A_B', header=None)
chem_df = df_raw.iloc[3:43, :5].copy()
chem_df.columns = ['No', 'Chemical', 'Source', 'Label_A', 'Label_B']
chem_df['No'] = chem_df['No'].astype(int)

print("Loading CACTUS molecular descriptors ...")
cactus = pd.read_csv(os.path.join(INPUT_DIR, 'cactus_features_40chem.csv'))
CACTUS_COLS = ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'QED',
               'BBB_penetrant', 'BBB_score', 'Brenk_alerts', 'PAINS_alerts']
cactus['compound_norm'] = cactus['compound'].str.lower().str.strip()
cactus_lookup = {row['compound_norm']: {c: row[c] for c in CACTUS_COLS}
                 for _, row in cactus.iterrows()}

# ── 2. Compound maps ──────────────────────────────────────────────────────
NO2PFX = {
    1:'Acitretin', 2:'Busulfan', 3:'Carbamazepine', 4:'Cyclophosphamide',
    5:'Fluconazole', 6:'5-Fluorouracil', 7:'Hydroxyurea', 8:'Isotretinoin',
    9:'Methotrexate', 10:'Phenytoin', 11:'Thalidomide', 12:'Topiramate',
    13:'Tretinoin', 14:'Valproic acid', 15:'Pomalidomide', 16:'Aspirin',
    17:'Cytarabine', 18:'Ibuprofen', 19:'Trimethadione', 20:'Vismodegib',
    21:'Ibrutinib', 22:'Pazopanib', 23:'Ribavirin', 24:'Tacrolimus',
    25:'Bosentan', 26:'Cisplatin', 27:'Dabrafenib', 28:'Dasatinib',
    29:'Imatinib', 30:'Rotenone', 31:'Acetaminophen', 32:'Acetylsalicylic acid',
    33:'D-Glucitol', 34:'L-Ascorbic acid', 35:'Saccharin Sodium Salt hydrate',
    36:'Deltamethrin', 37:"2,2',4,4',5-Pentabromodiphenyl ether",
    38:'Bisphenol A', 39:'Chlorpyrifos', 40:'imidacloprid',
}
CACTUS_NAME_MAP = {
    'Saccharin Sodium Salt hydrate': 'saccharin',
    'imidacloprid': 'imidacloprid',
}
LABELED_A = list(range(1, 36))
UNKNOWN_A  = list(range(36, 41))

label_lookup = {}
for _, row in chem_df.iterrows():
    try:
        label_lookup[int(row['No'])] = int(row['Label_A'])
    except (ValueError, TypeError):
        pass

def get_logfc_cols(no):
    p = NO2PFX[no]
    return sorted([c for c in logfc.columns
                   if c.startswith(p + '_') or c.startswith(p + ' ')])

def get_cactus_feat(no):
    key = CACTUS_NAME_MAP.get(NO2PFX[no], NO2PFX[no].lower())
    return cactus_lookup.get(key, {c: np.nan for c in CACTUS_COLS})

# ── 3. Gene sets ──────────────────────────────────────────────────────────
gs150_raw = pd.read_csv(os.path.join(INPUT_DIR, 'CytoHubba_Top150_GeneSet.csv'))
genes_150  = [g for g in gs150_raw['name'].tolist() if g in logfc.index]
genes_182  = [g for g in pd.read_csv(os.path.join(INPUT_DIR, 'GeneSet182.csv'))['Gene'].tolist()
               if g in logfc.index]
print(f"GeneSet1 (GS150): {len(genes_150)} genes")
print(f"GeneSet2 (GS182): {len(genes_182)} genes")

# ── 4. Build feature matrices ─────────────────────────────────────────────
def build_logfc_X(nos, genes):
    """RNA-seq logFC sample × gene DataFrame."""
    frames = []
    for no in nos:
        cols = get_logfc_cols(no)
        if not cols:
            raise ValueError(f"No logFC cols for No.{no} ({NO2PFX[no]})")
        sub = logfc.loc[genes, cols].T.copy()
        sub.index = [f"{NO2PFX[no]}_{c.split('_')[-1]}" for c in cols]
        frames.append(sub)
    return pd.concat(frames, axis=0)

def build_cactus_X(nos, genes):
    """RNA-seq + molecular descriptor DataFrame."""
    X_rna = build_logfc_X(nos, genes)
    cac_rows = []
    for no in nos:
        feat = get_cactus_feat(no)
        for _ in range(len(get_logfc_cols(no))):
            cac_rows.append(feat)
    cac_df = pd.DataFrame(cac_rows, index=X_rna.index)
    return pd.concat([X_rna, cac_df], axis=1)

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

def run_fold_eval(X_tr, y_tr, X_te, y_te):
    """Train/evaluate all models for one fold. Returns metrics including CM."""
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(X_tr)
    Xte_s = sc.transform(X_te)
    results = {}
    for name, clf in make_models().items():
        clf.fit(Xtr_s, y_tr)
        yp    = clf.predict(Xte_s)
        yprob = clf.predict_proba(Xte_s)[:, 1]
        try:
            auc = roc_auc_score(y_te, yprob)
        except Exception:
            auc = np.nan
        tn, fp, fn, tp = confusion_matrix(y_te, yp, labels=[0, 1]).ravel()
        results[name] = {
            'AUC':      auc,
            'Accuracy': accuracy_score(y_te, yp),
            'F1':       f1_score(y_te, yp, zero_division=0),
            'MCC':      matthews_corrcoef(y_te, yp),
            'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
            'y_pred':   yp,
            'y_prob':   yprob,
        }
    return results

# ── 6. Save fold files ────────────────────────────────────────────────────
def save_fold_files(X_tr, y_tr, X_te, y_te, fold_dir):
    os.makedirs(fold_dir, exist_ok=True)
    X_tr.to_csv(os.path.join(fold_dir, 'input_data1.csv'))
    pd.DataFrame({'Category': y_tr}).to_csv(
        os.path.join(fold_dir, 'output_data1.csv'), index=False)
    X_te.to_csv(os.path.join(fold_dir, 'input_data2.csv'))
    pd.DataFrame({'Category': y_te}).to_csv(
        os.path.join(fold_dir, 'output_data2.csv'), index=False)

# ── 7. Core CV function ───────────────────────────────────────────────────
def run_cv_analysis(nos, valid_lbls, X_full, y_full, gs_name, analysis_type, out_dir):
    """
    5-fold compound-level StratifiedShuffleSplit cross-validation.

    Parameters
    ----------
    nos        : list of compound numbers (1..35)
    valid_lbls : label per compound (aligned with nos)
    X_full     : sample × feature DataFrame (rows indexed by sample name)
    y_full     : sample-level label array
    gs_name    : e.g. 'GeneSet150'
    analysis_type : 'Non-CACTUS' or 'CACTUS'
    out_dir    : output directory for this analysis
    """
    chem_lbl_arr = np.array(valid_lbls)

    # Map sample row → compound number
    idx_to_no = {}
    for no in nos:
        pfx = NO2PFX[no]
        for s in X_full.index:
            if s.startswith(pfx + '_') or s.startswith(pfx + ' '):
                idx_to_no[s] = no

    cv_dir = os.path.join(out_dir, 'CV_folds')
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=42)

    # Accumulate fold metrics
    fold_vals = {m: {k: [] for k in ['AUC', 'Accuracy', 'F1', 'MCC',
                                      'TP', 'TN', 'FP', 'FN']}
                 for m in ['LR', 'SVM', 'RF']}

    for fi, (tr_cidx, te_cidx) in enumerate(
            sss.split(np.zeros(len(nos)), chem_lbl_arr), start=1):

        tr_nos = {nos[i] for i in tr_cidx}
        te_nos = {nos[i] for i in te_cidx}

        tr_mask = np.array([idx_to_no.get(s, -1) in tr_nos for s in X_full.index])
        te_mask = np.array([idx_to_no.get(s, -1) in te_nos for s in X_full.index])

        X_tr, y_tr = X_full[tr_mask], y_full[tr_mask]
        X_te, y_te = X_full[te_mask], y_full[te_mask]

        save_fold_files(X_tr, y_tr, X_te, y_te, os.path.join(cv_dir, f'fold{fi}'))

        pos_tr = int(y_tr.sum())
        pos_te = int(y_te.sum())
        print(f"  Fold {fi}: train={len(tr_nos)} cpds ({pos_tr}pos/{len(y_tr)-pos_tr}neg)  "
              f"test={len(te_nos)} cpds ({pos_te}pos/{len(y_te)-pos_te}neg)")

        fold_res = run_fold_eval(X_tr.values, y_tr, X_te.values, y_te)
        for m, d in fold_res.items():
            for k in fold_vals[m]:
                fold_vals[m][k].append(d[k])

    # Build and save CV_metrics.csv
    cv_rows = []
    print(f"\n  {'Model':5s} {'Metric':10s}  Fold1   Fold2   Fold3   Fold4   Fold5   Mean±SD")
    for m in ['LR', 'SVM', 'RF']:
        for metric in ['AUC', 'Accuracy', 'F1', 'MCC', 'TP', 'TN', 'FP', 'FN']:
            vals = fold_vals[m][metric]
            mn = float(np.nanmean(vals))
            sd = float(np.nanstd(vals))
            cv_rows.append({
                'Label': 'A', 'GeneSet': gs_name,
                'Analysis': analysis_type, 'Model': m, 'Metric': metric,
                **{f'Fold{i+1}': vals[i] for i in range(5)},
                'Mean': mn, 'SD': sd,
            })
        for metric in ['AUC', 'Accuracy', 'F1', 'MCC']:
            vals = fold_vals[m][metric]
            mn, sd = np.nanmean(vals), np.nanstd(vals)
            fs = '  '.join(f'{v:.3f}' for v in vals)
            print(f"  {m:5s} {metric:10s}  {fs}  {mn:.3f}±{sd:.3f}")

    cv_df = pd.DataFrame(cv_rows)
    csv_path = os.path.join(out_dir, 'CV_metrics.csv')
    cv_df.to_csv(csv_path, index=False)
    print(f"  → CV_metrics.csv saved: {csv_path}")
    return cv_df

# ══════════════════════════════════════════════════════════════════════════
# BUILD COMPOUND-LEVEL LABELS
# ══════════════════════════════════════════════════════════════════════════
nos_A  = LABELED_A
lbls_A = [label_lookup[no] for no in nos_A]
y_A    = np.array([label_lookup[no] for no in nos_A
                   for _ in range(len(get_logfc_cols(no)))])

# ══════════════════════════════════════════════════════════════════════════
# ANALYSIS 1: Transcriptomics-based (Non-CACTUS) GeneSet1 (150 genes)
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("  ANALYSIS 1: Transcriptomics-based / GeneSet1 (150 genes)")
print(f"{'='*65}")

out1 = os.path.join(OUTPUT_DIR, 'NonCACTUS_GeneSet150')
X_nc150 = build_logfc_X(LABELED_A, genes_150)
run_cv_analysis(nos_A, lbls_A, X_nc150, y_A, 'GeneSet150', 'Non-CACTUS', out1)

# Unknown compound predictions (Transcriptomics / GS1)
X_nc150_unk = build_logfc_X(UNKNOWN_A, genes_150)
sc = StandardScaler()
Xtr_s  = sc.fit_transform(X_nc150.values)
Xunk_s = sc.transform(X_nc150_unk.values)
probs  = {}
for name, clf in make_models().items():
    clf.fit(Xtr_s, y_A)
    probs[name] = clf.predict_proba(Xunk_s)[:, 1]

unk_chem = [NO2PFX[no] for no in UNKNOWN_A for _ in range(len(get_logfc_cols(no)))]
pred_df = pd.DataFrame({'Chemical': unk_chem, **probs}, index=X_nc150_unk.index)
pred_df['Ensemble']   = pred_df[['LR', 'SVM', 'RF']].mean(axis=1)
pred_df['Prediction'] = (pred_df['Ensemble'] >= 0.5).astype(int)
fin_dir1 = os.path.join(out1, 'Final_prediction')
os.makedirs(fin_dir1, exist_ok=True)
pred_df.to_csv(os.path.join(fin_dir1, 'predictions.csv'))
X_nc150.to_csv(os.path.join(fin_dir1, 'input_data1.csv'))
X_nc150_unk.to_csv(os.path.join(fin_dir1, 'input_data2.csv'))
pd.DataFrame({'Category': y_A}).to_csv(os.path.join(fin_dir1, 'output_data1.csv'), index=False)
print(f"  → Final predictions saved: {fin_dir1}")

# ══════════════════════════════════════════════════════════════════════════
# ANALYSIS 2: Multi-modal (CACTUS) GeneSet2 (182 genes)
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("  ANALYSIS 2: Multi-modal (CACTUS) / GeneSet2 (182 genes)")
print(f"{'='*65}")

out2 = os.path.join(OUTPUT_DIR, 'CACTUS_GeneSet182')
X_cac182 = build_cactus_X(LABELED_A, genes_182)
run_cv_analysis(nos_A, lbls_A, X_cac182, y_A, 'GeneSet182', 'CACTUS', out2)

# Save merged feature file
X_save = X_cac182.copy(); X_save['Label_A'] = y_A
X_save.to_csv(os.path.join(OUTPUT_DIR, 'A_GeneSet182_with_cactus.csv'))

# ══════════════════════════════════════════════════════════════════════════
# ANALYSIS 3: Multi-modal (CACTUS) GeneSet1 (150 genes)
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("  ANALYSIS 3: Multi-modal (CACTUS) / GeneSet1 (150 genes)")
print(f"{'='*65}")

out3 = os.path.join(OUTPUT_DIR, 'CACTUS_GeneSet150')
X_cac150 = build_cactus_X(LABELED_A, genes_150)
run_cv_analysis(nos_A, lbls_A, X_cac150, y_A, 'GeneSet150', 'CACTUS', out3)

X_save2 = X_cac150.copy(); X_save2['Label_A'] = y_A
X_save2.to_csv(os.path.join(OUTPUT_DIR, 'A_GeneSet150_with_cactus.csv'))

# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("  ALL ANALYSES COMPLETE")
print(f"{'='*65}")
for label, d in [('Non-CACTUS GS150', out1), ('CACTUS GS182', out2), ('CACTUS GS150', out3)]:
    df = pd.read_csv(os.path.join(d, 'CV_metrics.csv'))
    print(f"\n{label}:")
    for m in ['LR', 'SVM', 'RF']:
        sub = df[(df['Model'] == m) & (df['Metric'].isin(['AUC', 'Accuracy', 'F1', 'MCC']))]
        vals = {r['Metric']: f"{r['Mean']:.3f}±{r['SD']:.3f}" for _, r in sub.iterrows()}
        print(f"  {m}: AUC={vals.get('AUC','?')}  Acc={vals.get('Accuracy','?')}  "
              f"F1={vals.get('F1','?')}  MCC={vals.get('MCC','?')}")
