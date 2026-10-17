# Multi-Sigma neural-network evaluation outputs (multi-modal model)

`GeneSet{1,2,3}/*_CV{1-5}_ROC.html` are the report pages exported by Multi-Sigma (AIZOTH Inc.) for the
five chemical-level evaluation splits of each gene set (Gene Set 1 = 182 genes, Gene Set 2 = 149 matched
genes, Gene Set 3 = 190 genes). Each page contains the figures `Category_violine`, `Category_roc` and
`Category_confusionMatrix` (embedded PNG images) of the held-out chemicals; the split definitions are in
`data/multisigma_inputs/*_CV_folds/` and the final-model settings in
`data/multisigma_inputs/final_training_results/`.
