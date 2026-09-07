"""
Export ML (LR/SVM/RF ensemble) SHAP-based feature contributions in the same
format as the user's own NN factor-analysis output (e.g. CACTUS_NN_GeneSet150.csv):

    Category,,,
    ,Contribution [%],Contribution_positive [%],Contribution_negative [%]
    <Feature>,<total%>,<positive%>,<negative%>
    ...

One file per combination (CACTUS x GeneSet182/150, Non-CACTUS x GeneSet182/150).

Methodology
-----------
All three classifiers' SHAP values must be expressed on the SAME scale
(predicted probability of Category 1) before they can be averaged into a
single "ML ensemble" contribution, matching how the final prediction itself
is the mean of the three classifiers' predict_proba outputs:

  - RF  : shap.TreeExplainer -> already in probability space for sklearn
          RandomForestClassifier (verified: base_value + sum(SHAP) == predict_proba).
  - SVM : shap.KernelExplainer on predict_proba -> probability space by construction.
  - LR  : shap.LinearExplainer explains the MARGIN (log-odds), not probability
          (verified empirically). To make it additive/comparable with RF and
          SVM, LR is instead explained with the same KernelExplainer-on-
          predict_proba approach used for SVM.

Ensemble SHAP per (sample, feature) = mean(SHAP_LR_proba, SHAP_RF_proba, SHAP_SVM_proba).
This is exact in expectation because Ensemble probability = mean(p_LR, p_RF, p_SVM)
and Shapley-value decomposition is linear in the explained function.

For each feature:
  pos_sum   = sum over the 105 samples of max(SHAP_ens, 0)   (push toward Category 1)
  neg_sum   = sum over the 105 samples of max(-SHAP_ens, 0)  (push toward Category 0)
  total     = pos_sum + neg_sum
Contribution[%]          = 100 * total   / sum_over_features(total)
Contribution_positive[%] = 100 * pos_sum / sum_over_features(total)
Contribution_negative[%] = 100 * neg_sum / sum_over_features(total)

CHECKPOINTED: re-run this script repeatedly (each call is time-boxed) until it
prints "ALL COMPLETE" -- background processes do not survive between calls in
this sandbox, so progress is checkpointed to shap_cache/ on disk.
"""
import os, time, pickle, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
import shap

OUTPUTS = '/sessions/elegant-beautiful-bell/mnt/outputs'
CACHE   = f'{OUTPUTS}/shap_cache'
EXPDIR  = f'{OUTPUTS}/ML_Contribution_Export'
os.makedirs(EXPDIR, exist_ok=True)

TIME_BUDGET = 33.0
t0 = time.time()
def elapsed(): return time.time() - t0
def log(msg): print(f'[{elapsed():7.1f}s] {msg}', flush=True)
def time_left(): return TIME_BUDGET - elapsed()

TAGS = ['CACTUS_GeneSet182', 'CACTUS_GeneSet150', 'NonCACTUS_GeneSet182', 'NonCACTUS_GeneSet150']
FILE_NAMES = {
    'CACTUS_GeneSet182':    'CACTUS_ML_GeneSet182.csv',
    'CACTUS_GeneSet150':    'CACTUS_ML_GeneSet150.csv',
    'NonCACTUS_GeneSet182': 'NonCACTUS_ML_GeneSet182.csv',
    'NonCACTUS_GeneSet150': 'NonCACTUS_ML_GeneSet150.csv',
}

with open(f'{CACHE}/setup_all.pkl', 'rb') as f:
    combos = pickle.load(f)


def kernel_proba_incremental(tag, c, model_key):
    """Incremental KernelExplainer on predict_proba for LR or SVM. Checkpointed."""
    state_path = f'{CACHE}/{tag}_{model_key}_proba_state.pkl'
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

    f_model = lambda x: models[model_key].predict_proba(x)[:, 1]
    expl = shap.KernelExplainer(f_model, state['background'])

    CHUNK = 8
    while state['cursor'] < n and time_left() > 4.0:
        end = min(state['cursor'] + CHUNK, n)
        rows = X_scaled.values[state['cursor']:end]
        sv_chunk = expl.shap_values(rows, nsamples=300, silent=True)
        state['sv'][state['cursor']:end] = sv_chunk
        state['cursor'] = end
        log(f'{tag}: {model_key}(proba) SHAP rows {state["cursor"]}/{n} done '
            f'(time_left={time_left():.1f}s)')
        with open(state_path, 'wb') as f:
            pickle.dump(state, f)

    if state['cursor'] >= n:
        log(f'{tag}: {model_key}(proba) SHAP fully complete.')
        return state['sv']
    return None


def get_rf_proba(tag, c):
    """RF TreeExplainer is already in probability space; reuse cached LR/RF pickle."""
    path = f'{CACHE}/{tag}_lr_rf.pkl'
    if os.path.exists(path):
        with open(path, 'rb') as f:
            d = pickle.load(f)
        return d['sv_rf']
    expl_rf = shap.TreeExplainer(c['models']['RF'])
    sv_rf_raw = expl_rf.shap_values(c['X_scaled'].values)
    if isinstance(sv_rf_raw, list):
        sv_rf = sv_rf_raw[1]
    elif sv_rf_raw.ndim == 3:
        sv_rf = sv_rf_raw[:, :, 1]
    else:
        sv_rf = sv_rf_raw
    return sv_rf


def get_svm_proba(tag, c):
    """SVM KernelExplainer (already computed for the Excel deliverable); reuse if complete."""
    state_path = f'{CACHE}/{tag}_svm_state.pkl'
    if os.path.exists(state_path):
        with open(state_path, 'rb') as f:
            state = pickle.load(f)
        if state['cursor'] >= len(c['X_scaled']):
            return state['sv']
    return kernel_proba_incremental(tag, c, 'SVM')


def write_nn_format_csv(tag, c, sv_ens):
    feat_names = c['feat_names']
    pos_sum = np.maximum(sv_ens, 0).sum(axis=0)
    neg_sum = np.maximum(-sv_ens, 0).sum(axis=0)
    total = pos_sum + neg_sum
    grand_total = total.sum()

    df = pd.DataFrame({
        'Feature': feat_names,
        'Contribution': 100 * total / grand_total,
        'Contribution_positive': 100 * pos_sum / grand_total,
        'Contribution_negative': 100 * neg_sum / grand_total,
    }).sort_values('Contribution', ascending=False).reset_index(drop=True)

    out_path = f'{EXPDIR}/{FILE_NAMES[tag]}'
    with open(out_path, 'w') as f:
        f.write('Category,,,\n')
        f.write(',Contribution [%],Contribution_positive [%],Contribution_negative [%]\n')
        for _, row in df.iterrows():
            f.write(f"{row['Feature']},{row['Contribution']:.2f},"
                    f"{row['Contribution_positive']:.2f},{row['Contribution_negative']:.2f}\n")
    log(f'{tag}: exported -> {out_path}')


def main():
    for tag in TAGS:
        out_path = f'{EXPDIR}/{FILE_NAMES[tag]}'
        if os.path.exists(out_path):
            log(f'{tag}: already exported, skipping.')
            continue

        c = combos[tag]

        sv_rf = get_rf_proba(tag, c)

        sv_svm = get_svm_proba(tag, c)
        if sv_svm is None:
            log(f'{tag}: SVM(proba) not complete; resume next call.')
            return

        sv_lr = kernel_proba_incremental(tag, c, 'LR')
        if sv_lr is None:
            log(f'{tag}: LR(proba) not complete; resume next call.')
            return

        sv_ens = (sv_lr + sv_rf + sv_svm) / 3.0
        write_nn_format_csv(tag, c, sv_ens)

    log('ALL COMPLETE.')


if __name__ == '__main__':
    main()
