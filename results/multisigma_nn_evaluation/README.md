# Multi-Sigma neural-network evaluation outputs

Report pages exported by Multi-Sigma (AIZOTH Inc.) for the five chemical-level evaluation splits of each gene set
(Gene Set 1 = 182 genes, Gene Set 2 = 149 matched genes, Gene Set 3 = 190 genes):

- `multimodal/GeneSet*/` — multi-modal model (genes + 10 descriptors; re-run on 13 Oct 2026 after the No. 32 descriptor correction)
- `transcriptomics/GeneSet*/` — transcriptomics-only model (14 Jun 2026 for Gene Sets 1 and 2, 21 Jul 2026 for Gene Set 3; no descriptors, hence not affected by the correction)

Each page contains the figures `Category_violine`, `Category_roc` and `Category_confusionMatrix` (embedded PNG images) of the held-out
chemicals. Multi-Sigma does not name the splits in the transcriptomics pages; CV1-CV5 follow the order in which the pages were exported,
which is the order of the splits. Split definitions: `data/multisigma_inputs/*_CV_folds/` (the transcriptomics-only inputs are these
files without the 10 descriptor columns); final-model settings: `data/multisigma_inputs/final_training_results/`.
