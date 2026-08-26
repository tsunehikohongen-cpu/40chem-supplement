# 40chem-supplement

Supplementary analysis code and result tables for the 40-chemical
transcriptomics manuscript: a heterogeneity-quantification figure (GO BP
NES profile analysis) and the ML/NN factor-analysis (feature contribution)
results referenced in the Supplement.

Companion repository (main figure scripts): [40chem-figures](https://github.com/tsunehikohongen-cpu/40chem-figures)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tested with Python 3.12.

## Contents

### `scripts/S1_heterogeneity.py` — Supplementary Figure (heterogeneity)
Quantifies heterogeneity of ssGSEA NES profiles across the full
pre-selection GO BP term set (2,692 terms x 40 chemicals), to show that the
30 development-related pathways used in Figure 2B are not an artifact of
picking an unusually variable term subset. Three panels:

1. Violin plot of pairwise NES-profile correlations for the 35 labeled
   chemicals, split into within-category vs. between-category pairs
   (Mann-Whitney U test).
2. Compound x compound NES-profile correlation heatmap
   (Category1 | Category0 | Unknown order).
3. Histogram of per-term cross-compound NES variance, with keyword-selected
   and ontology-selected developmental-term medians overlaid.

**Inputs:**
- `--nes` — the ssGSEA NES matrix (terms x 40 chemicals), the *same* matrix
  produced by `2b_Pathway_Heatmap_GO_BP_Development_reproduce.py` in
  [40chem-figures](https://github.com/tsunehikohongen-cpu/40chem-figures)
  (`output/ssgsea_NES_GO_Biological_Process_2023.csv`). Term names must
  carry their GO id, e.g. `"... (GO:0001234)"`. Column order must be fixed:
  Category1 (20 chemicals) | Category0 (15) | Unknown (5).
- `--keywords` — `data/S1_keyword_list.csv` (included in this repo):
  columns `[role, keyword_or_root]`, `role` one of `include` / `exclude` /
  `ontology_root`.
- `--obo` — `go-basic.obo` (Gene Ontology), not included; download once:
  ```bash
  curl -L -o go-basic.obo https://current.geneontology.org/ontology/go-basic.obo
  ```

**Run:**
```bash
python scripts/S1_heterogeneity.py \
  --nes path/to/ssgsea_NES_GO_Biological_Process_2023.csv \
  --keywords data/S1_keyword_list.csv \
  --obo go-basic.obo \
  --out output/SX_heterogeneity.png
```

**Output:** `SX_heterogeneity.png` (600 dpi by default; `--dpi` to change),
plus console output of term counts, the within/between correlation medians
and Mann-Whitney p-value, and the variance medians (all terms vs. ontology-
vs. keyword-selected subsets).

### `scripts/run_shap_contribution_export.py` — ML SHAP feature-contribution export
Computes, per gene-set/CACTUS-status combination, each feature's percent
contribution to the ML ensemble's (LR + SVM + RF) predicted probability of
Category 1, using SHAP values placed on a common (probability) scale and
averaged across the three classifiers. Output format matches the NN
factor-analysis CSVs in `results/factor_analysis/` so both are directly
comparable.

**This script cannot be run standalone in this repository.** It consumes a
pre-built `setup_all.pkl` cache (pickled `{tag: {X_scaled, models,
feat_names}}`, containing the fitted LR/RF/SVM models and scaled feature
matrix for each combination) produced by the original model-training
pipeline, which is not archived here. It is included for methodological
transparency — to document exactly how the `Contribution` /
`Contribution_positive` / `Contribution_negative` percentages in
`results/factor_analysis/CACTUS_ML_*.csv` and `NonCACTUS_ML_*.csv` were
computed — not as a turnkey pipeline. See the script's module docstring for
the full method (Shapley-value linearity argument for averaging SHAP across
classifiers) and the exact `setup_all.pkl` schema it expects.

Because `shap.KernelExplainer` estimation is expensive, the script is
time-boxed per invocation (`--time-budget`, default 33 s) and checkpoints
progress to `--cache-dir`; re-run repeatedly until it prints "ALL COMPLETE".

### `results/factor_analysis/` — feature-contribution result tables
Per-feature contribution percentages (`Contribution [%]`,
`Contribution_positive [%]`, `Contribution_negative [%]`) for each of the 4
combinations (CACTUS / Non-CACTUS status x GeneSet182 / GeneSet150 gene
sets), for both modeling approaches:

- `CACTUS_ML_GeneSet182.csv`, `CACTUS_ML_GeneSet150.csv`,
  `NonCACTUS_ML_GeneSet182.csv`, `NonCACTUS_ML_GeneSet150.csv` — ML
  (LR/SVM/RF ensemble) SHAP-based contributions, produced by
  `run_shap_contribution_export.py`.
- `CACTUS_NN_GeneSet150.csv`, `CACTUS_NN_GenSet182_要因分析.csv`,
  `Non-CACTUS_NN_GeneSet150.csv`, `Non-CACTUS_NN_GeneSet182.csv` — NN
  factor-analysis contributions (produced by a separate NN pipeline, not
  included in this repository).

## License

See `LICENSE` (MIT) for the code. The result tables in
`results/factor_analysis/` are aggregated summary statistics (per-feature
contribution percentages), not raw experimental data.

## Citing

See `CITATION.cff`.
