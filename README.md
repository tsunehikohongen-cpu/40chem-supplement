# 40chem-supplement

Supplementary data and analysis code for the 40-chemical iPSC transcriptomics
manuscript (mechanism-of-action structure of human iPSC responses and
developmental-toxicity prediction).

Companion repository (main-text figure scripts, expression matrix):
[40chem-figures](https://github.com/tsunehikohongen-cpu/40chem-figures)

## Which file belongs to which Supplementary item?

| Supplementary item | File(s) in this repository |
|---|---|
| Supplementary Fig. 1 — NES heterogeneity over all 2,692 GO BP terms | `scripts/Supplementary_Fig_1_NES_heterogeneity.py` |
| Supplementary Fig. 2 — PCA of RDKit molecular descriptors | `scripts/Supplementary_Fig_2_RDKit_PCA.py`, `data/Supplementary_Fig_2_10_descriptors_40chemicals.csv`, `data/Supplementary_Fig_2_RDKit_descriptors_40chemicals.csv` |
| Supplementary Table 1 — keyword / ontology-root definitions | `data/Supplementary_Table_1_keyword_list.csv` |
| Supplementary Table 2 — predictive performance of all model configurations | `scripts/ml_pipeline/` (cross-validation and metric tables) |
| Supplementary Table 4 / Supplementary Data 1 — ssGSEA leading-edge genes of the 30 developmental GO BP terms | `data/Supplementary_Data_1_leading_edge.xlsx`, `data/Supplementary_Data_1_leading_edge_per_gene.csv`, `scripts/Supplementary_Data_1_leading_edge.py` |
| Supplementary Table 5 — the 168 RDKit molecular descriptors | `data/Supplementary_Table_5_RDKit_descriptors.csv`, `scripts/Supplementary_Table_5_RDKit_descriptors.py` |
| Shared input: ssGSEA NES matrix (2,692 GO BP terms x 40 chemicals) | `data/NES_matrix_GO_BP_2023_40chemicals.csv` |

Supplementary Tables 3, 6 and 7 are given in the Supplementary Information file itself.

## Repository layout

```
data/
  Supplementary_Table_1_keyword_list.csv           Supplementary Table 1
  Supplementary_Data_1_leading_edge.xlsx           Supplementary Data 1 (README, LeadingEdge_summary, per_gene sheets)
  Supplementary_Data_1_leading_edge_per_gene.csv   Supplementary Data 1, per_gene sheet as a plain CSV
  Supplementary_Table_5_RDKit_descriptors.csv      Supplementary Table 5 (No., RDKit descriptor name)
  NES_matrix_GO_BP_2023_40chemicals.csv            ssGSEA NES matrix used by the scripts (same file as in 40chem-figures)
scripts/
  Supplementary_Fig_1_NES_heterogeneity.py
  Supplementary_Data_1_leading_edge.py
  Supplementary_Table_5_RDKit_descriptors.py
  fig_style.py                                     shared matplotlib style
  ml_pipeline/                                     model training / cross-validation pipeline (see below)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tested with Python 3.12. Run all scripts from the repository root; outputs are
written to `output/` (git-ignored).

## Data files

- **`NES_matrix_GO_BP_2023_40chemicals.csv`** — single-sample GSEA (ssGSEA)
  normalized enrichment scores of the 40 chemicals (columns, fixed order:
  Category 1 (20) | Category 0 (15) | Unknown (5)) for the 2,692 terms of the
  Enrichr library `GO_Biological_Process_2023` (rows). It is the same file as
  `data/NES_matrix_GO_BP_2023_40chemicals.csv` in
  [40chem-figures](https://github.com/tsunehikohongen-cpu/40chem-figures), where
  `scripts/compute_ssGSEA_NES_matrix.py` regenerates it from the expression
  matrix. SHA-256: `e49c67fac312cc33453840b2a16d742e591f1b739419fa561ec224c5e13a98aa`.
- **`Supplementary_Table_1_keyword_list.csv`** — columns `[role, keyword_or_root]`;
  `role` is `include` (41 keywords), `exclude` (6) or `ontology_root` (GO:0032502).
- **`Supplementary_Data_1_leading_edge*.{xlsx,csv}`** — for every chemical x
  developmental GO BP term of the 30-term heatmap (Supplementary Fig. 1a; formerly
  Fig. 2b of the main text), the leading-edge genes with their rank in the
  delta-rlog-ordered gene list, delta-rlog and percent contribution to the
  enrichment score. `per_gene` columns: `chemical, term, gene, rank,
  delta_rlog, contribution`. The workbook's `README` sheet documents the
  definitions; `LeadingEdge_summary` has one row per chemical x term.

## Scripts

### `scripts/Supplementary_Fig_1_NES_heterogeneity.py` — Supplementary Fig. 1
Three panels: (a) heatmap of the 30 developmental GO BP terms with the largest
cross-chemical NES variance among the 298 keyword-matched terms (Ward /
Euclidean row clustering); (b) 40 x 40 compound correlation heatmap
(Category 1 | Category 0 | Unknown); (c) histogram of per-term cross-chemical
NES variance with keyword-selected and ontology-selected subset medians.

- Inputs: `data/NES_matrix_GO_BP_2023_40chemicals.csv`,
  `data/Supplementary_Table_1_keyword_list.csv`, and `go-basic.obo`
  (Gene Ontology, not included; download once):
  ```bash
  curl -L -o go-basic.obo https://current.geneontology.org/ontology/go-basic.obo
  ```
- Run: `python scripts/Supplementary_Fig_1_NES_heterogeneity.py`
- Output: `output/Supplementary_Fig_1_NES_heterogeneity.{png,pdf}` (600 dpi)

### `scripts/Supplementary_Data_1_leading_edge.py` — Supplementary Data 1
Re-walks the delta-rlog-ranked gene list of every chemical for each of the 30
developmental terms (ssGSEA running sum, hit weight = |delta-rlog|^0.25),
locates the peak and extracts the leading-edge genes.

- Inputs: `40_Chem_DESeq_normalized.csv` (expression matrix; shipped in
  [40chem-figures](https://github.com/tsunehikohongen-cpu/40chem-figures)`/data/`;
  copy it to `data/` here), `data/NES_matrix_GO_BP_2023_40chemicals.csv`,
  `data/Supplementary_Table_1_keyword_list.csv`. `gseapy` downloads the
  `GO_Biological_Process_2023` library from Enrichr (internet required).
- Run: `python scripts/Supplementary_Data_1_leading_edge.py`
- Output: `output/Supplementary_Data_1_leading_edge.xlsx` and
  `output/Supplementary_Data_1_leading_edge_per_gene.csv` (the files in `data/`
  are the versions deposited with the manuscript).
- Note: genes with exactly identical delta-rlog values (e.g. non-expressed genes)
  have no defined order, so a re-run can differ from the deposited file in the
  rank/label of such tied genes (about 2.5% of rows) and in the ES at peak
  (by <= 0.004); delta-rlog and contribution values, gene-set sizes and
  leading-edge sizes are identical.

### `scripts/Supplementary_Table_5_RDKit_descriptors.py` — Supplementary Table 5
Checks that each of the 168 descriptor names of `data/Supplementary_Table_5_RDKit_descriptors.csv`
exists in `rdkit.Chem.Descriptors` of the installed RDKit and writes the list to
`output/Supplementary_Table_5_RDKit_descriptors.csv` (needs `rdkit`).

- Run: `python scripts/Supplementary_Table_5_RDKit_descriptors.py`

### `scripts/ml_pipeline/` — model training and evaluation pipeline
The end-to-end pipeline behind the classifier results (Supplementary Table 2),
archived as used for the manuscript. Several scripts expect manually prepared
inputs in `./inputs` and write to `./outputs` (see each script's header);
this is an archive for transparency rather than a one-command reproduction.

- `compute_cactus_features.py` — CACTUS-equivalent molecular descriptors (LogP,
  TPSA, MW, QED, BBB heuristic, HBD/HBA, Brenk/PAINS alerts) for the 40
  chemicals via PubChem + RDKit. No. 32 is
  5-acetylsalicylic acid (SMILES `CC(=O)C1=CC(=C(C=C1)O)C(=O)O`), a different
  compound from aspirin (No. 16); its SMILES is given explicitly so that a name search
  cannot resolve it to aspirin.
- `run_gs1_gs2_analysis.py`, `run_gs3_analysis.py` — LR / SVM / RF training and
  cross-validation (StratifiedShuffleSplit, compound-level) for Gene Set 1
  (182 genes), Gene Set 2 (150 genes) and Gene Set 3 (190 genes), in the
  transcriptomics-only (Non-CACTUS) and multi-modal (CACTUS) settings. Note: the
  internal labels in `run_gs1_gs2_analysis.py` ("GeneSet1 (Top150)", "GeneSet2
  (Top182)") are the reverse of the manuscript numbering used here (Gene Set 1 =
  182-gene Wilcoxon set, Gene Set 2 = 150-gene INGOR set).
- `build_gs190_excel.py`, `build_integrated_excel_v6.py` — formatted Excel
  summaries of the cross-validation metrics.
- `requirements.txt` — pinned dependency versions for this pipeline (differs
  from the top-level `requirements.txt`).

## License

See `LICENSE` (MIT) for the code. The data files are derived, aggregated results
(NES values and per-gene leading-edge statistics); raw RNA-seq data are deposited
in GEO under GSE345109.

## Citing

See `CITATION.cff` and the manuscript.
