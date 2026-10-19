# 40chem-supplement

Supplementary data and analysis code for the 40-chemical iPSC transcriptomics manuscript (mechanism-of-action
structure of human iPSC responses and developmental-toxicity prediction).

Companion repository (main-text figure scripts, expression matrix, Fig. 4 predictions, per-feature contributions):
[40chem-figures](https://github.com/tsunehikohongen-cpu/40chem-figures)

## Which file belongs to which Supplementary item?

| Supplementary item | File(s) in this repository |
|---|---|
| Fig. 1 — NES heterogeneity (2,692 GO BP terms) | `scripts/Supplementary_Fig_1_NES_heterogeneity.py` |
| Fig. 2 — PCA of RDKit descriptors | `scripts/Supplementary_Fig_2_RDKit_PCA.py`, `data/Supplementary_Fig_2_10_descriptors_40chemicals.csv`, `data/Supplementary_Fig_2_RDKit_descriptors_40chemicals.csv` |
| Table 1 — keyword / ontology-root definitions | `data/Supplementary_Table_1_keyword_list.csv` |
| Table 2 — performance of all model configurations | `data/supplementary_tables/Supplementary_Table_2_model_performance.csv`; ML rows recomputable from `data/multisigma_inputs/*_CV_folds/` (see below); NN: `results/multisigma_nn_evaluation/{multimodal,transcriptomics}/`; before/after the No. 32 descriptor correction: `data/Supplementary_Table_2_ML_multimodal_corrected.csv` |
| Table 3 — exposure conditions | `data/supplementary_tables/Supplementary_Table_3_exposure_conditions.csv` |
| Table 4 — leading-edge term overview | `data/supplementary_tables/Supplementary_Table_4_leading_edge_term_overview.csv` |
| Data 1 — gene-level leading-edge attribution | `data/Supplementary_Data_1_leading_edge.xlsx`, `data/Supplementary_Data_1_leading_edge_per_gene.csv`, `scripts/Supplementary_Data_1_leading_edge.py` |
| Table 5 — the 168 RDKit descriptors | `data/Supplementary_Table_5_RDKit_descriptors.csv`, `scripts/Supplementary_Table_5_RDKit_descriptors.py` |
| Table 6 — MoA vs ChEMBL / AOP-Wiki | `data/supplementary_tables/Supplementary_Table_6_MoA_ChEMBL_AOP.csv` |
| Table 7 — permutation tests (Fig. 5) | `data/supplementary_tables/Supplementary_Table_7_permutation_tests.csv`; code: `scripts/Fig5_MoA_permutation_test.py` in 40chem-figures |
| Table 8 — CAS RN, PubChem CID, SMILES, InChIKey, descriptors | `data/Master_chemical_descriptors_40chemicals.csv` |
| Table 9 — chemical-level performance | `data/supplementary_tables/Supplementary_Table_9_chemical_level_performance.csv`, `results/leakage_sensitivity/chem_level_folds.csv`, `scripts/ml_pipeline/chem_level_metrics.py` |
| Table 10 — Gene Set 1 re-selected within each split | `data/supplementary_tables/Supplementary_Table_10_nested_gene_selection.csv`, `results/leakage_sensitivity/nested_folds.csv`, `scripts/ml_pipeline/nested_gene_selection.py`, `data/nested_selection_inputs/` |
| Shared input: ssGSEA NES matrix | `data/NES_matrix_GO_BP_2023_40chemicals.csv` (identical to the file in 40chem-figures) |

The `data/supplementary_tables/*.csv` files are the tables of the Supplementary Information in machine-readable form
(numbers as printed there).

## Repository layout

```
data/
  supplementary_tables/        SI tables 2, 3, 4, 6, 7, 9, 10 as CSV
  multisigma_inputs/           model inputs: 5 chemical-level splits and the final training set for Gene Sets 1/2/3
                               (CACTUS_GeneSet{182,150,190}_*; numeric-only; *_rowkey.csv maps rows to samples),
                               final_training_results/ (settings of the 10 final networks per gene set)
  nested_selection_inputs/     inputs of the within-split Gene Set 1 re-selection (log fold changes, INGOR tables)
  Master_chemical_descriptors_40chemicals.csv, Supplementary_*.csv/xlsx, NES_matrix_*.csv
results/
  leakage_sensitivity/         per-split results of Supplementary Tables 9 and 10
  multisigma_nn_evaluation/    Multi-Sigma report pages of the evaluation splits (multi-modal NN)
scripts/
  Supplementary_*.py, fig_style.py
  ml_pipeline/                 model training and evaluation scripts
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tested with Python 3.12. Run scripts from the repository root unless stated otherwise; outputs go to `output/` (git-ignored).
`go-basic.obo` (Gene Ontology, needed by Supplementary Fig. 1) is not included:
`curl -L -o go-basic.obo https://current.geneontology.org/ontology/go-basic.obo`.
`40_Chem_DESeq_normalized.csv` (needed by Supplementary Data 1) is in 40chem-figures/data; copy it to `data/`.

## Reproducing the supplementary results

| Item | Command | Result |
|---|---|---|
| Supp. Fig. 1 | `python scripts/Supplementary_Fig_1_NES_heterogeneity.py` | `output/Supplementary_Fig_1_NES_heterogeneity.{png,pdf}` |
| Supp. Fig. 2 | `python scripts/Supplementary_Fig_2_RDKit_PCA.py` | PC1/PC2: (a) 52.0 / 21.9 %, (b) 27.0 / 11.0 % |
| Supp. Data 1 | `python scripts/Supplementary_Data_1_leading_edge.py` | same values; genes with identical delta-rlog have no defined order, so rank/label of such tied genes (about 2.5 % of rows) can differ on re-run |
| Supp. Table 5 | `python scripts/Supplementary_Table_5_RDKit_descriptors.py` | checks the 168 names against RDKit |
| Supp. Table 10 | `cd scripts/ml_pipeline && python nested_gene_selection.py` | LR and SVM identical to `results/leakage_sensitivity/nested_folds.csv`; RF can differ slightly (random forest, platform-dependent) |
| Supp. Tables 2 and 9 (ML models) | fit LR / SVM / RF (settings in `scripts/ml_pipeline/chem_level_metrics.py`) on `input_data1.csv`, predict `input_data2.csv` of each `data/multisigma_inputs/CACTUS_GeneSet{182,150,190}_CV_folds/fold{1-5}/`; for the transcriptomics-only model drop the 10 descriptor columns (MW, LogP, TPSA, HBD, HBA, QED, BBB_penetrant, BBB_score, Brenk_alerts, PAINS_alerts) | reproduces the ML values of Table 2 (102 of 108 values to 3 decimals; the others are random-forest differences) and of Table 9 |

Final-model ML probabilities (Fig. 4) are reproduced the same way from
`data/multisigma_inputs/CACTUS_GeneSet{182,150}_Final_prediction/` (training set `input_data1.csv`, the five Unknown chemicals
`input_data2.csv`; LR and SVM identical to the values in 40chem-figures, RF within 0.005).

## Neural-network analyses (Multi-Sigma)

The neural networks were built with Multi-Sigma (Lightstone Corp.), a commercial graphical platform. Provided here are the input
files of every evaluation split and of the final models (`data/multisigma_inputs/`), the settings of the final networks
(`final_training_results/`), the evaluation report pages (`results/multisigma_nn_evaluation/`), and, in 40chem-figures, the
final-model outputs for the Unknown chemicals and the per-feature contributions. The per-sample scores of the evaluation splits
are not exported as tables by Multi-Sigma (only the report pages); the networks cannot be re-trained without the platform.

## `scripts/ml_pipeline/` — model training pipeline

Archived as used for the manuscript; several scripts expect manually prepared inputs in `./inputs` and write to `./outputs`
(see each script's header), so this is documentation rather than a one-command run. The numbering inside
`run_gs1_gs2_analysis.py` ("GeneSet1 = Top150", "GeneSet2 = Top182") is the reverse of the manuscript (Gene Set 1 = 182-gene
Wilcoxon set, Gene Set 2 = 150-gene INGOR set).
`compute_cactus_features.py` computes the ten descriptors (No. 32 is 5-acetylsalicylic acid, given by explicit SMILES so that a name
search cannot resolve it to aspirin); `run_gs1_gs2_analysis.py`, `run_gs3_analysis.py` train and evaluate LR / SVM / RF;
`chem_level_metrics.py` and `nested_gene_selection.py` produce Supplementary Tables 9 and 10; `build_*_excel*.py` format result tables.

## License and citation

MIT (`LICENSE`). The data files are derived, aggregated results; raw RNA-seq data are deposited in GEO (GSE345109; released to reviewers
on request). Citation: `CITATION.cff` and the manuscript.
