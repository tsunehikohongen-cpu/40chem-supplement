Multi-Sigma用(数値のみ版)
- input_data*.csv: 1行目=特徴量名のヘッダ、以降は数値のみ。サンプル名の列は削除済み。
- output_data*.csv: 1行目=Category、以降は 1/0 のみ。
- input_data*_rowkey.csv: 行番号とサンプル名(化合物_濃度)の対応表(Multi-Sigmaには投入しない)。
- 中身の数値・行順・分割は前回パッケージと同一。
- GeneSet190の遺伝子名 HGC6.3 は Multi-Sigma の列名制約(ピリオド不可)のため HGC6_3 に変更(値は不変)。
