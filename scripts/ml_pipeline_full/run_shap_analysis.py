"""
SHAP analysis for the ML ensemble (LR / SVM / RF) used in the unknown-chemical
prediction figures, for all 4 model/modality x gene-set combinations.

CHECKPOINTED VERSION: background processes get killed when the sandbox call
ends, so this script does a bounded amount of work per invocation (time
budget below) and saves progress to disk. Re-running the script resumes from
the last checkpoint. Call it repeatedly until it prints "ALL COMPLETE".
"""
import os, time, json, pickle, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import shap

UPLOADS = '/sessions/elegant-beautiful-bell/mnt/uploads'
OUTPUTS = '/sessions/elegant-beautiful-bell/mnt/outputs'
RESDIR  = f'{OUTPUTS}/shap_results'
CACHE   = f'{OUTPUTS}/shap_cache'
os.makedirs(RESDIR, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

TIME_BUDGET = 33.0   # seconds; stop and checkpoint before the 45s call limit
t0 = time.time()
def elapsed():
    return time.time() - t0
def log(msg):
    print(f'[{elapsed():7.1f}s] {msg}', flush=True)
def time_left():
    return TIME_BUDGET - elapsed()

CACTUS_COLS = ['MW','LogP','TPSA','HBD','HBA','QED',
               'BBB_penetrant','BBB_score','Brenk_alerts','PAINS_alerts']

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
CACTUS_NAME_MAP = {'Saccharin Sodium Salt hydrate': 'saccharin', 'imidacloprid': 'imidacloprid'}
LABELED_A = list(range(1, 36))


def setup_data():
    """Build the 4 (X_scaled, y, models, feat_names) tuples, once. Cached to disk."""
    setup_path = f'{CACHE}/setup_all.pkl'
    if os.path.exists(setup_path):
        with open(setup_path, 'rb') as f:
            return pickle.load(f)

    log('Loading logFC...')
    logfc = pd.read_csv(f'{UPLOADS}/40 Chem logFC-ac99b402.csv', index_col=0)

    log('Loading labels...')
    df_raw  = pd.read_excel(f'{UPLOADS}/Table1_labels_A_B.xlsx', sheet_name='Labels A_B', header=None)
    chem_df = df_raw.iloc[3:43, :5].copy()
    chem_df.columns = ['No','Chemical','Source','Label_A','Label_B']
    chem_df['No'] = chem_df['No'].astype(int)
    chem_df = chem_df.reset_index(drop=True)

    log('Loading CACTUS molecular descriptors...')
    cactus = pd.read_csv(f'{OUTPUTS}/cactus_features_40chem.csv')
    cactus['compound_norm'] = cactus['compound'].str.lower().str.strip()
    cactus_lookup = {row['compound_norm']: {c: row[c] for c in CACTUS_COLS}
                     for _, row in cactus.iterrows()}

    label_lookup = {}
    for _, row in chem_df.iterrows():
        try:
            label_lookup[int(row['No'])] = int(row['Label_A'])
        except (ValueError, TypeError):
            pass

    def get_logfc_cols(no):
        p = NO2PFX[no]
        return sorted([c for c in logfc.columns if c.startswith(p + '_') or c.startswith(p + ' ')])

    def get_cactus_feat(no):
        key = CACTUS_NAME_MAP.get(NO2PFX[no], NO2PFX[no].lower())
        return cactus_lookup.get(key, {c: np.nan for c in CACTUS_COLS})

    gs150_raw = pd.read_csv(f'{UPLOADS}/CytoHubba_Top150_GeneSet.csv')
    genes_150 = [g for g in gs150_raw['name'].tolist() if g in logfc.index]
    genes_182 = [g for g in pd.read_csv(f'{UPLOADS}/GeneSet182.csv')['Gene'].tolist() if g in logfc.index]
    log(f'GS150: {len(genes_150)} genes, GS182: {len(genes_182)} genes')

    def build_logfc_X(nos, genes):
        frames = []
        for no in nos:
            cols = get_logfc_cols(no)
            sub = logfc.loc[genes, cols].T.copy()
            tok_map = {c: f"{NO2PFX[no]}_{c.split('_')[-1]}" for c in cols}
            sub.index = [tok_map[c] for c in cols]
            frames.append(sub)
        return pd.concat(frames, axis=0)

    def build_cactus_X(nos, genes):
        X_rna = build_logfc_X(nos, genes)
        cac_rows = []
        for no in nos:
            feat = get_cactus_feat(no)
            for _ in range(len(get_logfc_cols(no))):
                cac_rows.append(feat)
        cac_df = pd.DataFrame(cac_rows, index=X_rna.index)
        return pd.concat([X_rna, cac_df], axis=1)

    def make_models():
        return {
            'LR':  LogisticRegression(C=0.1, max_iter=2000, class_weight='balanced', random_state=42),
            'SVM': SVC(kernel='rbf', C=1.0, probability=True, class_weight='balanced', random_state=42),
            'RF':  RandomForestClassifier(n_estimators=200, class_weight='balanced_subsample', random_state=42),
        }

    y_full = np.array([label_lookup[no] for no in LABELED_A for _ in range(len(get_logfc_cols(no)))])

    combo_defs = [
        ('CACTUS',    'GeneSet182', build_cactus_X(LABELED_A, genes_182)),
        ('CACTUS',    'GeneSet150', build_cactus_X(LABELED_A, genes_150)),
        ('NonCACTUS', 'GeneSet182', build_logfc_X(LABELED_A, genes_182)),
        ('NonCACTUS', 'GeneSet150', build_logfc_X(LABELED_A, genes_150)),
    ]

    combos = {}
    for modality, gs, X_df in combo_defs:
        tag = f'{modality}_{gs}'
        sc = StandardScaler()
        X_scaled = pd.DataFrame(sc.fit_transform(X_df.values), index=X_df.index, columns=X_df.columns)
        models = make_models()
        for name, clf in models.items():
            clf.fit(X_scaled.values, y_full)
        feat_names = X_df.columns.tolist()
        is_descriptor = {f: (f in CACTUS_COLS) for f in feat_names}
        combos[tag] = dict(X_scaled=X_scaled, y=y_full, models=models,
                            feat_names=feat_names, is_descriptor=is_descriptor)
        log(f'{tag}: built X={X_scaled.shape}, models fit (LR/SVM/RF)')

    with open(setup_path, 'wb') as f:
        pickle.dump(combos, f)
    log('setup_all.pkl cached.')
    return combos


def lr_rf_results(tag, c):
    """Fast exact SHAP for LR + RF. Cached per-tag."""
    path = f'{CACHE}/{tag}_lr_rf.pkl'
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    X_scaled, models = c['X_scaled'], c['models']

    expl_lr = shap.LinearExplainer(models['LR'], X_scaled.values)
    sv_lr = expl_lr.shap_values(X_scaled.values)

    expl_rf = shap.TreeExplainer(models['RF'])
    sv_rf_raw = expl_rf.shap_values(X_scaled.values)
    if isinstance(sv_rf_raw, list):
        sv_rf = sv_rf_raw[1]
    elif sv_rf_raw.ndim == 3:
        sv_rf = sv_rf_raw[:, :, 1]
    else:
        sv_rf = sv_rf_raw

    result = {'sv_lr': sv_lr, 'sv_rf': sv_rf}
    with open(path, 'wb') as f:
        pickle.dump(result, f)
    log(f'{tag}: LR+RF SHAP cached.')
    return result


def svm_results_incremental(tag, c):
    """
    Incremental KernelExplainer for SVM. Returns None if not finished within
    the time budget (state checkpointed to disk); returns the SHAP array once
    every row has been explained.
    """
    state_path = f'{CACHE}/{tag}_svm_state.pkl'
    X_scaled, models = c['X_scaled'], c['models']
    n = len(X_scaled)

    if os.path.exists(state_path):
        with open(state_path, 'rb') as f:
            state = pickle.load(f)
    else:
        background = shap.sample(X_scaled, min(25, n), random_state=42)
        state = {'sv': np.full((n, X_scaled.shape[1]), np.nan), 'cursor': 0,
                  'background': background}

    if state['cursor'] >= n:
        return state['sv']

    f_svm = lambda x: models['SVM'].predict_proba(x)[:, 1]
    expl_svm = shap.KernelExplainer(f_svm, state['background'])

    CHUNK = 8
    while state['cursor'] < n and time_left() > 4.0:
        end = min(state['cursor'] + CHUNK, n)
        rows = X_scaled.values[state['cursor']:end]
        sv_chunk = expl_svm.shap_values(rows, nsamples=300, silent=True)
        state['sv'][state['cursor']:end] = sv_chunk
        state['cursor'] = end
        log(f'{tag}: SVM SHAP rows {state["cursor"]}/{n} done '
            f'(time_left={time_left():.1f}s)')
        with open(state_path, 'wb') as f:
            pickle.dump(state, f)

    if state['cursor'] >= n:
        log(f'{tag}: SVM SHAP fully complete.')
        return state['sv']
    return None


def assemble_and_save(tag, c, sv_lr, sv_rf, sv_svm):
    feat_names, is_descriptor = c['feat_names'], c['is_descriptor']
    out = pd.DataFrame({'Feature': feat_names,
                         'FeatureType': ['Descriptor' if is_descriptor[f] else 'Gene' for f in feat_names]})
    out['MeanAbsSHAP_LR']  = np.abs(sv_lr).mean(axis=0)
    out['MeanSHAP_LR']     = sv_lr.mean(axis=0)
    out['MeanAbsSHAP_RF']  = np.abs(sv_rf).mean(axis=0)
    out['MeanSHAP_RF']     = sv_rf.mean(axis=0)
    out['MeanAbsSHAP_SVM'] = np.abs(sv_svm).mean(axis=0)
    out['MeanSHAP_SVM']    = sv_svm.mean(axis=0)

    for m in ['LR', 'RF', 'SVM']:
        col = f'MeanAbsSHAP_{m}'
        mx = out[col].max()
        out[f'_norm_{m}'] = out[col] / mx if mx > 0 else 0.0
    out['Ensemble_Importance'] = out[['_norm_LR', '_norm_RF', '_norm_SVM']].mean(axis=1)
    out = out.drop(columns=['_norm_LR', '_norm_RF', '_norm_SVM'])
    out = out.sort_values('Ensemble_Importance', ascending=False).reset_index(drop=True)
    out.insert(0, 'Rank', np.arange(1, len(out) + 1))

    out_path = f'{RESDIR}/{tag}.csv'
    out.to_csv(out_path, index=False)
    log(f'{tag}: FINAL result saved -> {out_path}')


def main():
    combos = setup_data()
    tags = list(combos.keys())

    for tag in tags:
        final_path = f'{RESDIR}/{tag}.csv'
        if os.path.exists(final_path):
            log(f'{tag}: already complete, skipping.')
            continue

        c = combos[tag]
        lr_rf = lr_rf_results(tag, c)
        if time_left() < 4.0:
            log('Time budget exhausted after LR/RF step; resume next call.')
            return

        sv_svm = svm_results_incremental(tag, c)
        if sv_svm is None:
            log(f'{tag}: SVM SHAP not yet complete; resume next call.')
            return

        assemble_and_save(tag, c, lr_rf['sv_lr'], lr_rf['sv_rf'], sv_svm)

    log('ALL COMPLETE.')


if __name__ == '__main__':
    main()
