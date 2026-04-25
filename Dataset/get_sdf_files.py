import os
import json
import numpy as np
import requests
from urllib.parse import quote

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.inchi import MolToInchiKey
from rdkit.Chem import inchi


def smiles_to_3d_sdf_keep_stereo(
    smiles: str,
    sdf_path: str,
    max_tries: int = 20,
    seed0: int = 42,
    optimize: str = "MMFF",   # "MMFF" / "UFF" / None
):
    mol0 = Chem.MolFromSmiles(smiles)
    if mol0 is None:
        raise ValueError(f"Bad SMILES: {smiles}")

    # 输入的“规范 isomeric SMILES”（作为立体一致性基准）
    smi_ref = Chem.MolToSmiles(mol0, isomericSmiles=True)

    last_err = None
    for i in range(max_tries):
        try:
            mol = Chem.AddHs(mol0)

            params = AllChem.ETKDGv3()
            params.randomSeed = seed0 + i
            params.enforceChirality = True
            params.useExpTorsionAnglePrefs = True
            params.useBasicKnowledge = True

            code = AllChem.EmbedMolecule(mol, params)
            if code != 0:
                raise RuntimeError(f"EmbedMolecule failed (code={code})")

            # 优化（可选）
            if optimize is not None:
                if optimize.upper() == "MMFF" and AllChem.MMFFHasAllMoleculeParams(mol):
                    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
                else:
                    AllChem.UFFOptimizeMolecule(mol, maxIters=500)

            # 立体一致性检查：去掉H再比 isomeric SMILES
            mol_noh = Chem.RemoveHs(mol)
            Chem.AssignStereochemistry(mol_noh, force=True, cleanIt=True)
            smi_now = Chem.MolToSmiles(mol_noh, isomericSmiles=True)

            if smi_now != smi_ref:
                raise RuntimeError(f"Stereo mismatch: ref={smi_ref} vs now={smi_now}")

            # 写 SDF（含 3D conformer）
            mol.SetProp("SMILES", smi_ref)
            mol.SetProp("InChIKey", MolToInchiKey(mol_noh))

            w = Chem.SDWriter(sdf_path)
            w.write(mol)
            w.close()
            return  # success

        except Exception as e:
            last_err = e

    raise RuntimeError(f"Failed to generate 3D SDF with stereo after {max_tries} tries. Last error: {last_err}")


def load_dataset(name):
    data_raw = np.loadtxt(f'./{name}.txt', dtype=str, comments=None)
    return data_raw


'''get sdf files according to smiles'''
store = False
for name in ['OH', 'SO4-', 'O3', '1O2', 'Fe(VI)', 'HClO']: #
    error_list = []
    process_info = []
    smiles = load_dataset(name)[:, 0]

    for smi in smiles:
        # print(f'Start to process {smi}')
        mol = Chem.MolFromSmiles(smi)
        inchi_key = inchi.MolToInchiKey(mol)
        if f"{inchi_key}.sdf" not in os.listdir('./SDF'):
            try:
                mol = Chem.MolFromSmiles(smi)
                inchi_key = inchi.MolToInchiKey(mol)
                smiles_to_3d_sdf_keep_stereo(smi, f"./SDF/files/{inchi_key}.sdf")
                print(f'Success to process {smi}, using rdkit.')
                process_info.append(f'Success to process {smi}, using rdkit.')
            except:
                print(f'{smi} caused error.')
                process_info.append(f'{smi} caused error.')
                error_list.append(smi)
        else:
            print(f'{smi} has been processed.')

    if store:
        np.savetxt(f'./SDF/error/{name}.txt', error_list, fmt='%s')

    np.savetxt(f'./SDF/process_info/{name}.txt', process_info, fmt='%s')

''' get sdf of the typical pollutants '''
smiles = ['Oc1ccccc1', 'CCNC1=NC(=NC(=N1)NC(C)C)Cl',
          'Cc1noc(NS(=O)(=O)c2ccc(N)cc2)c1', 'CN(C)c1ccc(/N=N/c2ccc(S(=O)(=O)O)cc2)cc1',
          'C=CC#N', 'CCOP(=S)(OCC)Oc1nc(Cl)c(Cl)cc1Cl', 'CC(C)(C)O', 'COc1c(Cl)cc(Cl)cc1Cl', 'CCC(=O)C',
          'C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(=O)O']
for smi in smiles:
    print(f'Start to process {smi}')
    mol = Chem.MolFromSmiles(smi)
    inchi_key = inchi.MolToInchiKey(mol)

    smiles_to_3d_sdf_keep_stereo(smi, f"./SDF/files/{inchi_key}.sdf")
