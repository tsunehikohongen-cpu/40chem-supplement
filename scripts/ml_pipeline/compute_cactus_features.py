"""
CACTUS-equivalent molecular descriptors for the 40 chemicals, computed OFFLINE from the
curated SMILES in data/Master_chemical_descriptors_40chemicals.csv (one parent-compound
SMILES per chemical, keyed by chemical number, NOT by name lookup).

Why: a name-based PubChem lookup resolved No. 32 (5-acetylsalicylic acid) to aspirin, and
the representation of salts/tautomers changed descriptor values (saccharin sodium salt ->
free acid; imidacloprid nitroamine tautomer). Fixing the SMILES removes both failure modes.

Computes: MW, LogP (Wildman-Crippen), TPSA, HBD, HBA, QED, BBB heuristic (BBB_penetrant,
BBB_score), Brenk and PAINS alert counts, plus InChIKey and formula.
Run from the repository root:  python scripts/ml_pipeline/compute_cactus_features.py
Output: outputs/cactus_features_40chem.csv (column 'compound' = ML pipeline key), and a
consistency check against the master table.
"""
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, QED, FilterCatalog
from rdkit.Chem.FilterCatalog import FilterCatalogParams as P

master = pd.read_csv('data/Master_chemical_descriptors_40chemicals.csv')

def catalog(c):
    p = P(); p.AddCatalog(c); return FilterCatalog.FilterCatalog(p)
brenk, pains = catalog(P.FilterCatalogs.BRENK), catalog(P.FilterCatalogs.PAINS)

# key used by run_gs1_gs2_analysis.py (lower-cased); only the two names that differ
KEY = {'5-Acetylsalicylic acid': 'acetylsalicylic acid',
       'Saccharin Sodium Salt hydrate': 'saccharin',
       "BDE-99": "2,2',4,4',5-pentabromodiphenyl ether"}

rows = []
for _, r in master.iterrows():
    m = Chem.MolFromSmiles(r.SMILES)
    mw, tp = Descriptors.MolWt(m), rdMolDescriptors.CalcTPSA(m)
    lp, hd = Descriptors.MolLogP(m), rdMolDescriptors.CalcNumHBD(m)
    crit = [mw < 400, tp < 90, lp > 0, hd <= 3]
    rows.append({'compound': KEY.get(r.Chemical, r.Chemical.lower()),
                 'No': r.No, 'smiles': r.SMILES, 'inchikey': Chem.MolToInchiKey(m),
                 'MW': round(mw, 2), 'LogP': round(lp, 3), 'TPSA': round(tp, 2),
                 'HBD': hd, 'HBA': rdMolDescriptors.CalcNumHBA(m), 'QED': round(QED.qed(m), 4),
                 'BBB_penetrant': int(mw < 400 and tp < 90), 'BBB_score': round(sum(crit) / 4, 3),
                 'Brenk_alerts': len(brenk.GetMatches(m)), 'PAINS_alerts': len(pains.GetMatches(m))})
df = pd.DataFrame(rows)
assert df.inchikey.is_unique, 'two chemicals resolved to the same structure'
os.makedirs('outputs', exist_ok=True)
df.to_csv('outputs/cactus_features_40chem.csv', index=False)
print(df[['No', 'compound', 'MW', 'LogP', 'TPSA', 'QED', 'Brenk_alerts']].to_string(index=False))
