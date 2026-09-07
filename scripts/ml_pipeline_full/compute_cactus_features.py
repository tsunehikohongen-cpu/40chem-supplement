"""
CACTUS-equivalent molecular descriptor calculation for 40 chemicals.
Computes: LogP, TPSA, MW, QED, BBB heuristic, HBD, HBA, Brenk/PAINS alert counts.
"""
import requests, json, warnings
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, QED, FilterCatalog
from rdkit.Chem.FilterCatalog import FilterCatalogParams
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------
# 1. PubChem SMILES retrieval (name→CID fallback for tricky names)
# ------------------------------------------------------------------
CHEMICALS = [
    ('Acitretin',              'name', 'Acitretin'),
    ('Busulfan',               'name', 'Busulfan'),
    ('Carbamazepine',          'name', 'Carbamazepine'),
    ('Cyclophosphamide',       'name', 'Cyclophosphamide'),
    ('Fluconazole',            'name', 'Fluconazole'),
    ('5-Fluorouracil',         'name', '5-Fluorouracil'),
    ('Hydroxyurea',            'name', 'Hydroxyurea'),
    ('Isotretinoin',           'name', 'Isotretinoin'),
    ('Methotrexate',           'name', 'Methotrexate'),
    ('Phenytoin',              'name', 'Phenytoin'),
    ('Thalidomide',            'name', 'Thalidomide'),
    ('Topiramate',             'name', 'Topiramate'),
    ('Tretinoin',              'name', 'Tretinoin'),
    ('Valproic acid',          'name', 'Valproic acid'),
    ('Pomalidomide',           'name', 'Pomalidomide'),
    ('Aspirin',                'name', 'Aspirin'),
    ('Cytarabine',             'name', 'Cytarabine'),
    ('Ibuprofen',              'name', 'Ibuprofen'),
    ('Trimethadione',          'cid',  '5573'),
    ('Vismodegib',             'cid',  '24776445'),
    ('Ibrutinib',              'name', 'Ibrutinib'),
    ('Pazopanib',              'name', 'Pazopanib'),
    ('Ribavirin',              'name', 'Ribavirin'),
    ('Tacrolimus',             'name', 'Tacrolimus'),
    ('Bosentan',               'name', 'Bosentan'),
    ('Cisplatin',              'name', 'Cisplatin'),
    ('Dabrafenib',             'name', 'Dabrafenib'),
    ('Dasatinib',              'name', 'Dasatinib'),
    ('Imatinib',               'name', 'Imatinib'),
    ('Acetylsalicylic acid',   'name', 'Aspirin'),        # duplicate, same as Aspirin
    ('D-Glucitol',             'cid',  '5780'),
    ('L-Ascorbic acid',        'name', 'L-Ascorbic acid'),
    ('Saccharin',              'name', 'Saccharin'),
    ('Deltamethrin',           'name', 'Deltamethrin'),
    ("2,2',4,4',5-Pentabromodiphenyl ether", 'cid', '91695'),
    ('Bisphenol A',            'cid',  '6623'),
    ('Chlorpyrifos',           'cid',  '2730'),
    ('Imidacloprid',           'cid',  '86418'),
    ('Acetaminophen',          'name', 'Acetaminophen'),
    ('Rotenone',               'name', 'Rotenone'),
]

URL_NAME = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/property/IsomericSMILES,CanonicalSMILES/JSON"
URL_CID  = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IsomericSMILES,CanonicalSMILES/JSON"

def get_smiles(method, val):
    try:
        if method == 'cid':
            r = requests.get(URL_CID.format(cid=val), timeout=10)
        else:
            r = requests.post(URL_NAME, data={'name': val}, timeout=10)
        if r.status_code == 200:
            p = r.json()['PropertyTable']['Properties'][0]
            return (p.get('IsomericSMILES') or p.get('CanonicalSMILES') or p.get('SMILES', ''),
                    p.get('CID'))
    except:
        pass
    return None, None

from concurrent.futures import ThreadPoolExecutor, as_completed
smiles_map = {}
with ThreadPoolExecutor(max_workers=10) as ex:
    futures = {ex.submit(get_smiles, method, val): name for name, method, val in CHEMICALS}
    for fut in as_completed(futures):
        name = futures[fut]
        smi, cid = fut.result()
        smiles_map[name] = (smi, cid)
        print(f"{'OK' if smi else 'FAIL'}: {name}")

# ------------------------------------------------------------------
# 2. RDKit descriptor calculation
# ------------------------------------------------------------------
params_brenk = FilterCatalogParams()
params_brenk.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
catalog_brenk = FilterCatalog.FilterCatalog(params_brenk)

params_pains = FilterCatalogParams()
params_pains.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
catalog_pains = FilterCatalog.FilterCatalog(params_pains)

def bbb_heuristic(mol):
    """
    Simple CNS MPO-like BBB penetrant score.
    BBB+ if MW<400, TPSA<90, cLogP>0, HBD<=3
    Returns binary (1=penetrant, 0=non-penetrant) + probability-like score 0-1
    """
    mw   = Descriptors.MolWt(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    logp = Descriptors.MolLogP(mol)
    hbd  = rdMolDescriptors.CalcNumHBD(mol)
    # score = fraction of criteria met (0-4) / 4
    criteria = [mw < 400, tpsa < 90, logp > 0, hbd <= 3]
    score = sum(criteria) / len(criteria)
    penetrant = int(mw < 400 and tpsa < 90)
    return penetrant, round(score, 3)

records = []
for name, method, val in CHEMICALS:
    smi, cid = smiles_map.get(name, (None, None))
    row = {'compound': name, 'pubchem_cid': cid, 'smiles': smi}

    if smi:
        mol = Chem.MolFromSmiles(smi)
    else:
        mol = None

    if mol is None:
        print(f"  [WARN] Cannot parse SMILES for {name}")
        row.update({'MW': None, 'LogP': None, 'TPSA': None,
                    'HBD': None, 'HBA': None, 'QED': None,
                    'BBB_penetrant': None, 'BBB_score': None,
                    'Brenk_alerts': None, 'PAINS_alerts': None})
    else:
        bbb_bin, bbb_sc = bbb_heuristic(mol)
        row.update({
            'MW':            round(Descriptors.MolWt(mol), 2),
            'LogP':          round(Descriptors.MolLogP(mol), 3),
            'TPSA':          round(rdMolDescriptors.CalcTPSA(mol), 2),
            'HBD':           rdMolDescriptors.CalcNumHBD(mol),
            'HBA':           rdMolDescriptors.CalcNumHBA(mol),
            'QED':           round(QED.qed(mol), 4),
            'BBB_penetrant': bbb_bin,
            'BBB_score':     bbb_sc,
            'Brenk_alerts':  len(catalog_brenk.GetMatches(mol)),
            'PAINS_alerts':  len(catalog_pains.GetMatches(mol)),
        })
    records.append(row)

df = pd.DataFrame(records)
out_path = '/sessions/elegant-beautiful-bell/mnt/outputs/cactus_features_40chem.csv'
df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print(f"\nFeature table preview:")
print(df[['compound','MW','LogP','TPSA','QED','BBB_penetrant','BBB_score','Brenk_alerts','PAINS_alerts']].to_string(index=False))
