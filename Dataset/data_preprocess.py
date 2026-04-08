# environment: PhD-RC; preprocess the combined data, including filter invalid smiles and repeated smiles
import numpy as np
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger
from collections import Counter


def smiles_standardize(smiles: str, basicClean=True, clearCharge=False,
                       clearFrag=True, canonTautomer=False, isomeric=True) -> str:
    RDLogger.DisableLog('rdApp.*')
    try:
        mol_cleaned = Chem.MolFromSmiles(smiles)
        # 除去氢、金属原子、标准化分子
        if basicClean:
            mol_cleaned = rdMolStandardize.Cleanup(mol_cleaned)
        if clearFrag:
            mol_cleaned = rdMolStandardize.FragmentParent(mol_cleaned)
        # 尝试中性化处理分子
        if clearCharge:
            un_charger = rdMolStandardize.Uncharger()
            mol_cleaned = un_charger.uncharge(mol_cleaned)
        # 处理互变异构情形，这一步在某些情况下可能不够完美
        if canonTautomer:
            te = rdMolStandardize.TautomerEnumerator()  # idem
            mol_cleaned = te.Canonicalize(mol_cleaned)
        # 移除立体信息，并将分子存为标准化后的SMILES形式
        standard_smiles = Chem.MolToSmiles(mol_cleaned, isomericSmiles=isomeric)
    except Exception as e:
        print(e, smiles)
        return None
    return standard_smiles


def data_clean(smiles_list, values):
    data = [[], []]
    count_dict = Counter(smiles_list)
    index = 0
    while index < (len(smiles_list) - 1):
        smiles = smiles_list[index]
        count = count_dict[smiles]
        values_list = [values[i] for i in range(index, index+count, 1)]
        if len(values_list) > 1:
            data[0].append(smiles)
            data[1].append((max(values_list) + min(values_list))/2)
            index = index + count
            continue
        else:
            data[0].append(smiles)
            data[1].append(values[index])
            index = index + count
    data = np.array(data).T.reshape(-1, 2)
    return data


for name in ['OH', 'SO4-', 'O3', 'Fe(VI)', 'HClO', '1O2']: # 'OH', 'SO4-', 'O3', 'Fe(VI)', 'HClO', 'O2-', '1O2'
    data = np.loadtxt(f'{name}-combination.txt', dtype=str, comments=None)
    smiles = data[:, 0]
    labels = data[:, 1]
    sources = data[:, 2]

    smiles_processed = []
    labels_processed = []
    sources_processed = []
    infos_processed = []
    for idx, smi in enumerate(smiles):
        if smi == 'None' or smiles_standardize(smi) == None:
            continue
        print(f'{smi} was converted to {smiles_standardize(smi)}')
        infos_processed.append(f'{smi}^-^was^-^converted^-^to^-^{smiles_standardize(smi)}')
        smiles_processed.append(smiles_standardize(smi))
        labels_processed.append(labels[idx])
        sources_processed.append(sources[idx])

    np.savetxt(f'{name}_smiles_clean.txt',  np.array([smiles_processed, labels_processed, sources_processed, infos_processed]).T, fmt='%s')



for name in ['OH', 'SO4-', 'O3', 'Fe(VI)', 'HClO', '1O2']: # 'OH', 'SO4-', 'O3', 'Fe(VI)', 'HClO', 'O2-'
    print(f'start to process {name}')
    data = np.loadtxt(f'{name}_smiles_clean.txt', dtype=str, comments=None)
    smiles = data[:, 0]
    labels = data[:, 1].astype(float)
    sources = data[:, 2]

    count_dict = {}
    for idx, smi in enumerate(smiles):
        if smi not in count_dict:
            count_dict[smi] = [[labels[idx]], [sources[idx]]]
        else:
            count_dict[smi][0].append(labels[idx])
            count_dict[smi][1].append(sources[idx])

    smiles_processed_2 = []
    labels_processed_2 = []
    infos_processed_2 = []
    for key, value in count_dict.items():
        smiles_processed_2.append(key)
        labels_processed_2.append(np.mean(value[0]))
        infos_processed_2.append(';'.join([str(np.round(v, 3)) for v in value[0]])+'::'+';'.join(value[1]))

    np.savetxt(f'{name}.txt',  np.array([smiles_processed_2, labels_processed_2, infos_processed_2]).T, fmt='%s')