import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
import json


def maccs(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return list(MACCSkeys.GenMACCSKeys(mol))


def morgan(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    return list(GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))


def rdkit(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return list(Chem.RDKFingerprint(mol))


def get_fingerprints(smiles_list, type):
    if type == 'rdkit':
        return list(map(rdkit, smiles_list))
    elif type == 'morgan':
        return list(map(morgan, smiles_list))
    elif type == 'maccs':
        return list(map(maccs, smiles_list))
    else:
        raise ValueError(f'Do not have {type} encoder')

type = 'rdkit'
for name in ['OH', 'SO4-', 'O3', '1O2', 'Fe(VI)', 'HClO']:
    data = np.loadtxt(f'../{name}.txt', dtype=str, comments=None)
    smiles_list = list(data[:, 0])
    emb = get_fingerprints(smiles_list, type)
    mappings_smiles_emb = dict(zip(smiles_list, emb))
    with open(f"{name}-{type}.json", "w", encoding="utf-8") as f:
        json.dump(mappings_smiles_emb, f, ensure_ascii=False, indent=2)

