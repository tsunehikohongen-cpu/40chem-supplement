#!/usr/bin/env python3
"""Supplementary Table 5 — Full list of 168 RDKit molecular descriptors used
in the extended-panel PCA (Supplementary Fig. 2).

The authoritative descriptor list is `data/Supplementary_Table_5_RDKit_descriptors.csv`
(168 rows). This script verifies that every descriptor name in the list exists in
`rdkit.Chem.Descriptors` under the installed RDKit, and re-emits the list with a short
per-descriptor description from each function's docstring.

Run from the repository root:  python scripts/Supplementary_Table_5_RDKit_descriptors.py

Output
------
- output/Supplementary_Table_5_RDKit_descriptors.csv
  (columns: No, RDKit_descriptor_name, short_description)
"""
import csv
import os
import pandas as pd

try:
    from rdkit.Chem import Descriptors as _D
    _func = {n: f for n, f in _D.descList}
except Exception:
    _func = {}

names = pd.read_csv('data/Supplementary_Table_5_RDKit_descriptors.csv')['RDKit_descriptor_name'].tolist()
assert len(names) == 168, f'expected 168, got {len(names)}'

rows = []
missing = []
for i, n in enumerate(names, 1):
    f = _func.get(n)
    if f is None:
        missing.append(n)
        doc = ''
    else:
        doc = (f.__doc__ or '').strip().split('\n')[0][:120]
    rows.append((i, n, doc))

os.makedirs('output', exist_ok=True)
with open('output/Supplementary_Table_5_RDKit_descriptors.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['No', 'RDKit_descriptor_name', 'short_description'])
    w.writerows(rows)

print(f'wrote output/Supplementary_Table_5_RDKit_descriptors.csv ({len(rows)} descriptors)')
if missing:
    print(f'  WARNING: {len(missing)} descriptor name(s) not found in installed RDKit:')
    for n in missing: print(f'    - {n}')
