# environment: unimol
from rdkit.Chem import PandasTools
from unimol_tools.data.datareader import MolDataReader
from unimol_tools.data.conformer import UniMolV2Feature, ConformerGen
from unimol_tools.predictor import MolDataset
from unimol_tools.tasks.trainer import Trainer
from unimol_tools.models.unimolv2 import UniMolV2Model
from unimol_tools.models.unimol import UniMolModel
import torch
import os
import numpy as np
import h5py
from rdkit import Chem
from rdkit.Chem import inchi
from tqdm import tqdm


def get_func(model_name):
    use_cuda = True  # params
    device = torch.device("cuda:0" if torch.cuda.is_available() and use_cuda else "cpu")
    if model_name == 'unimolv1':
        params = {'data_type': 'molecule', 'remove_hs': False, 'model_name': 'unimolv1',
                  'use_cuda': True, 'use_ddp': False, 'use_gpu': 'all', 'save_path': None}  # 'batch_size': 32,
        func_preprocess = lambda x: ConformerGen(**params).transform_mols(x)
        trainer = Trainer(task='repr', **params)
        model = UniMolModel(output_dim=1, data_type='molecule').to(device)

    elif model_name == 'unimolv2':
        params = {'data_type': 'molecule', 'remove_hs': False, 'model_name': 'unimolv2',
                  'model_size': '84m', 'use_cuda': True, 'use_ddp': False, 'use_gpu': 'all', 'save_path': None}
        func_preprocess = lambda x: UniMolV2Feature(**params).transform_mols(x)
        trainer = Trainer(task='repr', **params)
        model = UniMolV2Model(output_dim=1, model_size='84m').to(device)
    func_repr = lambda dataset: trainer.inference(model, return_repr=True, return_atomic_reprs=True, dataset=dataset)

    return func_preprocess, func_repr


def get_repr_from_smi(smiles=None, func_preprocess=None, func_repr=None):  # smi, func_preprocess, func_repr
    mol = Chem.MolFromSmiles(smiles)
    inchi_key = inchi.MolToInchiKey(mol)
    sdf_path = f'../../Dataset/SDF/files/{inchi_key}.sdf'
    data = PandasTools.LoadSDF(sdf_path)
    data = MolDataReader()._convert_numeric_columns(data)
    no_h_list = func_preprocess([data['ROMol'][0]])
    dataset = MolDataset(no_h_list)
    repr_output = func_repr(dataset)
    atom_reprs = repr_output["atomic_reprs"][0]
    return atom_reprs


def save_name_mats_h5(path, names, mats):
    assert len(names) == len(mats)
    with h5py.File(path, "w") as f:
        # 保存名字
        f.create_dataset("names", data=np.array(names, dtype="S"))  # bytes

        g = f.create_group("mats")
        for i, (name, mat) in enumerate(zip(names, mats)):
            mat = np.asarray(mat)  # 确保是 numpy
            key = f"{i:06d}_{name}"  # 防止重名冲突
            g.create_dataset(key, data=mat, compression="gzip", compression_opts=4)


def load_name_mats_h5(path):
    with h5py.File(path, "r") as f:
        names = [x.decode("utf-8") for x in f["names"][...]]
        g = f["mats"]
        mats = [g[f"{i:06d}_{name}"][...] for i, name in enumerate(names)]
    return names, mats


if __name__ == "__main__":
    model_name = 'unimolv2'
    func_preprocess, func_repr = get_func(model_name)
    smiles_list = ['Oc1ccccc1', 'Oc1ccc(Cl)cc1', 'CC(C)(c1ccc(O)cc1)c1ccc(O)cc1', 'CCNc1nc(Cl)nc(NC(C)C)n1',
              'CC1=CC(=NO1)NS(=O)(=O)C2=CC=C(C=C2)N', 'NC(=O)N1c2ccccc2C=Cc2ccccc21', 'CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O',
              'CN(C)c1ccc(cc1)N=NC1=CC=CC=C1S(=O)(=O)[O-]', 'ClC=C(Cl)Cl',
              'C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(=O)O']
    atom_reprs_all = []
    smiles_all = []
    name = 'typical'
    for smi in tqdm(list(smiles_list)):
        try:
            atom_reprs = get_repr_from_smi(smi, func_preprocess, func_repr)
            atom_reprs_all.append(atom_reprs)
            smiles_all.append(smi)
        except:
            print(smi)
    save_path = '../../Dataset/Embeddings' + os.sep + name + '-'+ model_name +'.h5'
    save_name_mats_h5(save_path, smiles_all, atom_reprs_all)
    raise ValueError




    name_list = ['OH', 'SO4-', 'O3', '1O2', 'Fe(VI)', 'HClO'] #
    model_name = 'unimolv2'
    func_preprocess, func_repr = get_func(model_name)

    for name in name_list:
        print(name)
        data = np.loadtxt(f'../../Dataset/{name}.txt', dtype=str, comments=None)
        smiles_list = data[:, 0]
        atom_reprs_all = []
        smiles_all = []
        for smi in tqdm(list(smiles_list)):
            try:
                atom_reprs = get_repr_from_smi(smi, func_preprocess, func_repr)
                atom_reprs_all.append(atom_reprs)
                smiles_all.append(smi)
            except:
                print(smi)

        save_path = '../../Dataset/Embeddings' + os.sep + name + '-'+ model_name +'.h5'
        save_name_mats_h5(save_path, smiles_all, atom_reprs_all)


    # name = 'HClO'
    # model_name = 'unimolv1'
    # path_emb = '/home/huatianwei/Code/PhD/Rate constant/Dataset/Embeddings' + os.sep + name + '-'+ model_name +'.h5'
    # smiles, embeddings = load_name_mats_h5(path_emb)
