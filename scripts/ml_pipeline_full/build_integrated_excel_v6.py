"""
Build LabelA_integrated_metrics.xlsx (v6, xlsxwriter)
- NN rows: BLANK except CACTUS GS150 (Gene Set 2 INGOR), which is filled from uploaded HTML
- CACTUS GS150 NN data: extracted from 5 ROC HTML files (per-fold AUC, Acc, F1, MCC)
"""
import os, math
import numpy as np
import pandas as pd
import xlsxwriter

OUTPUTS = '/sessions/elegant-beautiful-bell/mnt/outputs'

# ── Load CV results ───────────────────────────────────────────────────────────
def load_cv(path):
    df = pd.read_csv(path)
    out = {}
    for model in df['Model'].unique():
        sub = df[df['Model'] == model]
        out[model] = {}
        for _, row in sub.iterrows():
            metric = row['Metric']
            out[model][metric] = {
                'folds': [row[f'Fold{i}'] for i in range(1, 6)],
                'mean':  row['Mean'],
                'sd':    row['SD'],
            }
    return out

nc_gs182 = load_cv(f'{OUTPUTS}/ML_analysis/LabelA/GeneSet182/CV_metrics.csv')
nc_gs150 = load_cv(f'{OUTPUTS}/ML_analysis/LabelA/GeneSet150/CV_metrics.csv')
ca_gs182 = load_cv(f'{OUTPUTS}/LabelA/GeneSet182/CV_metrics.csv')
ca_gs150 = load_cv(f'{OUTPUTS}/LabelA/GeneSet150/CV_metrics.csv')

# ── CACTUS GS150 NN results (extracted from 5 ROC HTML files) ────────────────
# CM layout: top-left=FP, top-right=TP, bottom-left=TN, bottom-right=FN
# Verified: N_pos=30, N_neg=24, N=54 for all folds
_nn_cms = [
    # (FP, TP, TN, FN)
    (4, 19, 20, 11),   # fold 1 (roc_CBGm)
    (1, 28, 23,  2),   # fold 2 (roc_SJmw)
    (4, 21, 20,  9),   # fold 3 (roc_U4Ga)
    (3, 25, 21,  5),   # fold 4 (roc_Yr7m)
    (3, 30, 21,  0),   # fold 5 (roc_bO2z)
]
_nn_auc  = [0.743, 0.974, 0.807, 0.843, 0.954]

def _cm2metrics(fp, tp, tn, fn):
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    denom = ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**0.5
    mcc  = (tp * tn - fp * fn) / denom if denom > 0 else 0.0
    return acc, f1, mcc

_nn_acc, _nn_f1, _nn_mcc = zip(*[_cm2metrics(*cm) for cm in _nn_cms])
_nn_tp = [cm[1] for cm in _nn_cms]
_nn_tn = [cm[2] for cm in _nn_cms]
_nn_fp = [cm[0] for cm in _nn_cms]
_nn_fn = [cm[3] for cm in _nn_cms]

def _ms(vals):
    return np.mean(vals), np.std(vals, ddof=1)

CA_GS150_NN = {
    'AUC':      {'folds': list(_nn_auc),          'mean': _ms(_nn_auc)[0],  'sd': _ms(_nn_auc)[1]},
    'Accuracy': {'folds': list(_nn_acc),          'mean': _ms(_nn_acc)[0],  'sd': _ms(_nn_acc)[1]},
    'F1':       {'folds': list(_nn_f1),           'mean': _ms(_nn_f1)[0],   'sd': _ms(_nn_f1)[1]},
    'MCC':      {'folds': list(_nn_mcc),          'mean': _ms(_nn_mcc)[0],  'sd': _ms(_nn_mcc)[1]},
    'TP': list(_nn_tp), 'TN': list(_nn_tn),
    'FP': list(_nn_fp), 'FN': list(_nn_fn),
}

# ── CACTUS GS182 NN results (extracted from 5 ROC HTML files) ────────────────
# Layout: top-left=FP, top-right=TP, bottom-left=TN, bottom-right=FN
# Verified: N_pos=30, N_neg=24, N=54 for all folds
_nn182_cms = [
    (0, 28, 24, 2),   # fold 1 (roc_U30X)
    (1, 30, 23, 0),   # fold 2 (roc_aauV)
    (0, 22, 24, 8),   # fold 3 (roc_l7Zg)
    (2, 23, 22, 7),   # fold 4 (roc_m6Yn)
    (0, 27, 24, 3),   # fold 5 (roc_obEJ)
]
_nn182_auc = [0.990, 0.992, 0.897, 0.867, 0.972]

_nn182_acc, _nn182_f1, _nn182_mcc = zip(*[_cm2metrics(*cm) for cm in _nn182_cms])
_nn182_tp = [cm[1] for cm in _nn182_cms]
_nn182_tn = [cm[2] for cm in _nn182_cms]
_nn182_fp = [cm[0] for cm in _nn182_cms]
_nn182_fn = [cm[3] for cm in _nn182_cms]

CA_GS182_NN = {
    'AUC':      {'folds': list(_nn182_auc),   'mean': _ms(_nn182_auc)[0],  'sd': _ms(_nn182_auc)[1]},
    'Accuracy': {'folds': list(_nn182_acc),   'mean': _ms(_nn182_acc)[0],  'sd': _ms(_nn182_acc)[1]},
    'F1':       {'folds': list(_nn182_f1),    'mean': _ms(_nn182_f1)[0],   'sd': _ms(_nn182_f1)[1]},
    'MCC':      {'folds': list(_nn182_mcc),   'mean': _ms(_nn182_mcc)[0],  'sd': _ms(_nn182_mcc)[1]},
    'TP': list(_nn182_tp), 'TN': list(_nn182_tn),
    'FP': list(_nn182_fp), 'FN': list(_nn182_fn),
}

# ── Non-CACTUS GS182 NN results ──────────────────────────────────────────────
# Layout: top-left=FP, top-right=TP, bottom-left=TN, bottom-right=FN
# Verified: N_pos=30, N_neg=24, N=54 for all folds
_nc182_cms = [
    (2, 26, 22, 4),   # fold1 (roc_NAyU)
    (1, 22, 23, 8),   # fold2 (roc_V1yo)
    (1, 27, 23, 3),   # fold3 (roc_fT3i)
    (0, 27, 24, 3),   # fold4 (roc_fphe)
    (0, 27, 24, 3),   # fold5 (roc_nTlK)
]
_nc182_auc = [0.939, 0.861, 0.983, 0.956, 0.958]

_nc182_acc, _nc182_f1, _nc182_mcc = zip(*[_cm2metrics(*cm) for cm in _nc182_cms])
_nc182_tp = [cm[1] for cm in _nc182_cms]
_nc182_tn = [cm[2] for cm in _nc182_cms]
_nc182_fp = [cm[0] for cm in _nc182_cms]
_nc182_fn = [cm[3] for cm in _nc182_cms]

NC_GS182_NN = {
    'AUC':      {'folds': list(_nc182_auc),  'mean': _ms(_nc182_auc)[0],  'sd': _ms(_nc182_auc)[1]},
    'Accuracy': {'folds': list(_nc182_acc),  'mean': _ms(_nc182_acc)[0],  'sd': _ms(_nc182_acc)[1]},
    'F1':       {'folds': list(_nc182_f1),   'mean': _ms(_nc182_f1)[0],   'sd': _ms(_nc182_f1)[1]},
    'MCC':      {'folds': list(_nc182_mcc),  'mean': _ms(_nc182_mcc)[0],  'sd': _ms(_nc182_mcc)[1]},
    'TP': list(_nc182_tp), 'TN': list(_nc182_tn),
    'FP': list(_nc182_fp), 'FN': list(_nc182_fn),
}

# ── Non-CACTUS GS150 NN results ──────────────────────────────────────────────
# Layout: top-left=FP, top-right=TP, bottom-left=TN, bottom-right=FN
# Verified: N_pos=30, N_neg=24, N=54 for all folds
_nc150_cms = [
    (0, 25, 24, 5),   # fold1 (roc_JJeD)
    (2, 27, 22, 3),   # fold2 (roc_Pq2q)
    (4, 25, 20, 5),   # fold3 (roc_aoyf)
    (1, 23, 23, 7),   # fold4 (roc_phK5)
    (9, 25, 15, 5),   # fold5 (roc_uhZb)
]
_nc150_auc = [0.949, 0.951, 0.857, 0.893, 0.804]

_nc150_acc, _nc150_f1, _nc150_mcc = zip(*[_cm2metrics(*cm) for cm in _nc150_cms])
_nc150_tp = [cm[1] for cm in _nc150_cms]
_nc150_tn = [cm[2] for cm in _nc150_cms]
_nc150_fp = [cm[0] for cm in _nc150_cms]
_nc150_fn = [cm[3] for cm in _nc150_cms]

NC_GS150_NN = {
    'AUC':      {'folds': list(_nc150_auc),  'mean': _ms(_nc150_auc)[0],  'sd': _ms(_nc150_auc)[1]},
    'Accuracy': {'folds': list(_nc150_acc),  'mean': _ms(_nc150_acc)[0],  'sd': _ms(_nc150_acc)[1]},
    'F1':       {'folds': list(_nc150_f1),   'mean': _ms(_nc150_f1)[0],   'sd': _ms(_nc150_f1)[1]},
    'MCC':      {'folds': list(_nc150_mcc),  'mean': _ms(_nc150_mcc)[0],  'sd': _ms(_nc150_mcc)[1]},
    'TP': list(_nc150_tp), 'TN': list(_nc150_tn),
    'FP': list(_nc150_fp), 'FN': list(_nc150_fn),
}

# ── CM derivation for LR/SVM/RF ───────────────────────────────────────────────
N, N_POS, N_NEG = 54, 30, 24

def derive_cm(f1, acc):
    if abs(1.0 - f1) < 1e-9:
        return N_POS, N_NEG, 0, 0
    if abs(f1) < 1e-9:
        tn = round(acc * N); return 0, tn, N_NEG - tn, N_POS
    tp = round(f1 * N * (1 - acc) / (2 * (1 - f1)))
    tn = round(acc * N) - tp
    return tp, tn, N_NEG - tn, N_POS - tp

def mean_cm(cv_data, model):
    cms = [derive_cm(f1, acc) for f1, acc in
           zip(cv_data[model]['F1']['folds'], cv_data[model]['Accuracy']['folds'])]
    return tuple(np.mean([c[i] for c in cms]) for i in range(4))

def fmt_ms(m, s, d=3):
    f = f'{{:.{d}f}}'
    return f'{f.format(m)}±{f.format(s)}'

def ms(cv, model, metric):
    return fmt_ms(cv[model][metric]['mean'], cv[model][metric]['sd'])

# ── Workbook setup ─────────────────────────────────────────────────────────────
out_path = f'{OUTPUTS}/LabelA_integrated_metrics.xlsx'
wb = xlsxwriter.Workbook(out_path)

TNR = 'Times New Roman'

C_HDR   = '#4472C4'
C_FEAT  = '#D9E1F2'
C_GS1   = '#E2EFDA'
C_GS2   = '#FFF2CC'
C_NN    = '#F2F2F2'
C_WHITE = '#FFFFFF'

def fmt(bg=C_WHITE, bold=False, color='#000000', size=10,
        align='center', valign='vcenter', wrap=False, num_fmt=None):
    kw = dict(
        font_name=TNR, font_size=size, bold=bold, font_color=color,
        bg_color=bg, align=align, valign=valign, text_wrap=wrap,
        border=1, border_color='#000000',
    )
    if num_fmt:
        kw['num_format'] = num_fmt
    return wb.add_format(kw)

F_HDR  = fmt(bg=C_HDR,  bold=True, color='#FFFFFF', size=10)
F_FEAT = fmt(bg=C_FEAT, bold=True)
F_GS1  = fmt(bg=C_GS1,  bold=True)
F_GS2  = fmt(bg=C_GS2,  bold=True)
F_NN   = fmt(bg=C_NN)
F_NN_B = fmt(bg=C_NN, bold=True)
F_NORM = fmt()
F_BOLD = fmt(bold=True)
F_D4   = fmt(num_fmt='0.0000')
F_D4B  = fmt(bold=True, num_fmt='0.0000')
F_D1   = fmt(num_fmt='0.0')
F_NN_D4 = fmt(bg=C_NN, num_fmt='0.0000')
F_NN_D1 = fmt(bg=C_NN, num_fmt='0.0')

MODELS  = ['LR', 'SVM', 'RF', 'NN']
METRICS = ['AUC', 'Accuracy', 'F1', 'MCC']

# =============================================================================
# SHEET 1: Summary
# =============================================================================
ws = wb.add_worksheet('Summary')
ws.freeze_panes(1, 0)

cols_hdr = ["Feature Set","Gene Set","Model",
            "AUC","Accuracy","F1","MCC",
            "TP","TN","FP","FN"]
col_w    = [22, 20, 8, 16, 16, 16, 16, 8, 8, 8, 8]
for ci, (h, w) in enumerate(zip(cols_hdr, col_w)):
    ws.set_column(ci, ci, w)
    ws.write(0, ci, h, F_HDR)
ws.set_row(0, 24)

# (feat_label, gs_label, cv_data, gs_fmt, nn_data_or_None)
SECTIONS = [
    ("Multi-modal model",          "Gene Set 1 (Wilcoxon)", ca_gs182, F_GS1, CA_GS182_NN),
    ("Multi-modal model",          "Gene Set 2 (INGOR)",    ca_gs150, F_GS2, CA_GS150_NN),
    ("Transcriptomics-based model","Gene Set 1 (Wilcoxon)", nc_gs182, F_GS1, NC_GS182_NN),
    ("Transcriptomics-based model","Gene Set 2 (INGOR)",    nc_gs150, F_GS2, NC_GS150_NN),
]

r = 1
for feat_lbl, gs_lbl, cv, gs_fmt, nn_data in SECTIONS:
    n_models = len(MODELS)
    ws.merge_range(r, 0, r+n_models-1, 0, feat_lbl, F_FEAT)
    ws.merge_range(r, 1, r+n_models-1, 1, gs_lbl,   gs_fmt)
    for model in MODELS:
        is_nn = (model == 'NN')
        mf = F_NN_B if is_nn else F_BOLD
        ws.write(r, 2, model, mf)
        if is_nn:
            if nn_data is not None:
                # Fill NN row from provided data
                ws.write(r, 3,  fmt_ms(nn_data['AUC']['mean'],      nn_data['AUC']['sd']),      F_NN)
                ws.write(r, 4,  fmt_ms(nn_data['Accuracy']['mean'],  nn_data['Accuracy']['sd']),  F_NN)
                ws.write(r, 5,  fmt_ms(nn_data['F1']['mean'],        nn_data['F1']['sd']),        F_NN)
                ws.write(r, 6,  fmt_ms(nn_data['MCC']['mean'],       nn_data['MCC']['sd']),       F_NN)
                ws.write(r, 7,  round(np.mean(nn_data['TP']), 1), F_NN_D1)
                ws.write(r, 8,  round(np.mean(nn_data['TN']), 1), F_NN_D1)
                ws.write(r, 9,  round(np.mean(nn_data['FP']), 1), F_NN_D1)
                ws.write(r, 10, round(np.mean(nn_data['FN']), 1), F_NN_D1)
            else:
                for ci in range(3, 11):
                    ws.write(r, ci, '', F_NN)
        else:
            ws.write(r, 3,  ms(cv, model, 'AUC'),      F_NORM)
            ws.write(r, 4,  ms(cv, model, 'Accuracy'),  F_NORM)
            ws.write(r, 5,  ms(cv, model, 'F1'),        F_NORM)
            ws.write(r, 6,  ms(cv, model, 'MCC'),       F_NORM)
            tp, tn, fp, fn = mean_cm(cv, model)
            ws.write(r, 7,  round(tp, 1), F_D1)
            ws.write(r, 8,  round(tn, 1), F_D1)
            ws.write(r, 9,  round(fp, 1), F_D1)
            ws.write(r, 10, round(fn, 1), F_D1)
        r += 1

# =============================================================================
# SHEET 2 & 3: PerFold
# =============================================================================
def make_perfold(wb, sheet_name, gs_sections):
    """gs_sections: [(gs_lbl, cv_data, gs_fmt, nn_data_or_None)]"""
    ws = wb.add_worksheet(sheet_name)
    ws.freeze_panes(1, 0)

    hdr_cols = ["Gene Set","Model","Metric",
                "Fold 1","Fold 2","Fold 3","Fold 4","Fold 5",
                "Mean","SD"]
    hdr_w    = [20, 8, 10, 9, 9, 9, 9, 9, 12, 10]
    for ci, (h, w) in enumerate(zip(hdr_cols, hdr_w)):
        ws.set_column(ci, ci, w)
        ws.write(0, ci, h, F_HDR)
    ws.set_row(0, 22)

    r = 1
    for gs_lbl, cv, gs_fmt, nn_data in gs_sections:
        n_rows_gs = len(MODELS) * len(METRICS)
        ws.merge_range(r, 0, r+n_rows_gs-1, 0, gs_lbl, gs_fmt)
        for model in MODELS:
            is_nn = (model == 'NN')
            mf = F_NN_B if is_nn else F_BOLD
            ws.merge_range(r, 1, r+len(METRICS)-1, 1, model, mf)
            for metric in METRICS:
                ws.write(r, 2, metric, F_NN if is_nn else F_NORM)
                if is_nn:
                    if nn_data is not None:
                        d = nn_data[metric]
                        for fi, fv in enumerate(d['folds']):
                            ws.write(r, 3+fi, round(fv, 4), F_NN_D4)
                        ws.write(r, 8, round(d['mean'], 4), fmt(bg=C_NN, bold=True, num_fmt='0.0000'))
                        ws.write(r, 9, round(d['sd'],   4), F_NN_D4)
                    else:
                        for ci in range(3, 10):
                            ws.write(r, ci, '', F_NN)
                else:
                    d = cv[model][metric]
                    for fi, fv in enumerate(d['folds']):
                        ws.write(r, 3+fi, round(fv, 4), F_D4)
                    ws.write(r, 8, round(d['mean'], 4), F_D4B)
                    ws.write(r, 9, round(d['sd'],   4), F_D4)
                r += 1

make_perfold(wb, 'MultiModal_PerFold',
    [("Gene Set 1 (Wilcoxon)", ca_gs182, F_GS1, CA_GS182_NN),
     ("Gene Set 2 (INGOR)",    ca_gs150, F_GS2, CA_GS150_NN)])

make_perfold(wb, 'Transcriptomics_PerFold',
    [("Gene Set 1 (Wilcoxon)", nc_gs182, F_GS1, NC_GS182_NN),
     ("Gene Set 2 (INGOR)",    nc_gs150, F_GS2, NC_GS150_NN)])

# =============================================================================
# SHEET 4: Legend & Notes
# =============================================================================
ws4 = wb.add_worksheet('Legend & Notes')
ws4.set_column(0, 0, 30)
ws4.set_column(1, 1, 65)

notes = [
    ("Sheet", "Description", True),
    ("Summary",                  "Mean±SD across 5 folds; TP/TN/FP/FN = average of per-fold derived values", False),
    ("MultiModal_PerFold",       "Per-fold metrics for Multi-modal model (CACTUS; RNA-seq logFC + molecular descriptors)", False),
    ("Transcriptomics_PerFold",  "Per-fold metrics for Transcriptomics-based model (Non-CACTUS; RNA-seq logFC only)", False),
    ("", "", False),
    ("Term", "Definition", True),
    ("Multi-modal model",        "CACTUS: RNA-seq logFC + 10 molecular descriptors", False),
    ("Transcriptomics-based model","Non-CACTUS: RNA-seq logFC features only", False),
    ("Gene Set 1 (Wilcoxon)",    "182 DEGs selected by Wilcoxon test", False),
    ("Gene Set 2 (INGOR)",       "150 hub genes from INGOR network, CytoHubba MCC ranking (Top 150; 149 matched in logFC data)", False),
    ("CV method",                "StratifiedShuffleSplit (n_splits=5, test_size=0.5, random_state=42)", False),
    ("Splitting unit",           "Compound-level; all 3 concentration samples per compound masked together", False),
    ("N per fold (test set)",    "54 samples (30 positive, 24 negative)", False),
    ("NN (Multi-modal)",         "Neural Network — to be analyzed separately; all cells left blank intentionally", False),
    ("NN (CACTUS Gene Set 1)",   "NN results filled from AIZOTH ROC HTML output (5 folds)", False),
    ("NN (CACTUS Gene Set 2)",   "NN results filled from AIZOTH ROC HTML output (5 folds)", False),
    ("", "", False),
    ("Model", "Description", True),
    ("LR",  "Logistic Regression (C=0.1, max_iter=2000, class_weight=balanced)", False),
    ("SVM", "Support Vector Machine (RBF kernel, C=1.0, class_weight=balanced)", False),
    ("RF",  "Random Forest (n_estimators=200, class_weight=balanced_subsample)", False),
    ("NN",  "Neural Network (AIZOTH; CACTUS GS182 & GS150 results provided; Multi-modal TBD)", False),
]

for ri, (k, v, is_hdr) in enumerate(notes):
    kf = fmt(bg=C_HDR, bold=True, color='#FFFFFF') if is_hdr else fmt(align='left')
    vf = fmt(bg=C_HDR, bold=True, color='#FFFFFF') if is_hdr else fmt(align='left', wrap=True)
    ws4.write(ri, 0, k, kf)
    ws4.write(ri, 1, v, vf)
    ws4.set_row(ri, 18 if is_hdr else 15)

wb.close()
print(f"Saved: {out_path}")

# Print summary for verification
print("\n=== CACTUS GS150 NN Summary ===")
print(f"AUC:      {fmt_ms(CA_GS150_NN['AUC']['mean'], CA_GS150_NN['AUC']['sd'])}")
print(f"Accuracy: {fmt_ms(CA_GS150_NN['Accuracy']['mean'], CA_GS150_NN['Accuracy']['sd'])}")
print(f"F1:       {fmt_ms(CA_GS150_NN['F1']['mean'], CA_GS150_NN['F1']['sd'])}")
print(f"MCC:      {fmt_ms(CA_GS150_NN['MCC']['mean'], CA_GS150_NN['MCC']['sd'])}")
print(f"TP (mean): {round(np.mean(CA_GS150_NN['TP']), 1)}")
print(f"TN (mean): {round(np.mean(CA_GS150_NN['TN']), 1)}")
print(f"FP (mean): {round(np.mean(CA_GS150_NN['FP']), 1)}")
print(f"FN (mean): {round(np.mean(CA_GS150_NN['FN']), 1)}")
