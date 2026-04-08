# environment: unimol
import numpy as np
import pandas as pd


def k_fold_mask(number_sample, seed=0, k_fold=5):
    mask = [index for index in range(number_sample)]
    np.random.seed(seed)
    np.random.shuffle(mask)
    mask_list = []
    gap = number_sample // k_fold
    for index in range(k_fold):
        mask_temp = mask[index*gap:(index+1)*gap]
        mask_list.append(mask_temp)
    return mask_list


name_list = ['OH', 'SO4-', 'O3', '1O2', 'Fe(VI)', 'HClO']
for name in name_list:
    data = np.loadtxt(f'../../Dataset/{name}.txt', dtype=str, comments=None)
    smiles_list = data[:, 0]
    labels = data[:, 1].astype(float)

    mask_cv_fold = k_fold_mask(len(smiles_list), seed=0, k_fold=5)

    for idx, mask_test_fold in enumerate(mask_cv_fold):
        mask_train_fold = list(np.setdiff1d(np.arange(len(smiles_list)), np.array(mask_test_fold)))
        if name == 'OH':
            smiles_train_valid, labels_train_valid = [], []
            smiles_train = smiles_list[mask_train_fold]
            labels_train = labels[mask_train_fold]
            for idx_j, smi in enumerate(smiles_train):
                if smi in [r'[H]/N=C(\N)SCCCc1cnc[nH]1', 'C[C@H](N=C(O)CN=C(O)[C@H](CO)N=C(O)[C@H](CS)N=C(O)CN=C(O)C(CS)N=C(O)C(CC(=O)O)N=C(O)CN)C(O)=N[C@@H](CO)C(O)=N[C@@H](CO)C(O)=N[C@@H](CS)C(O)=N[C@H](C(O)=N[C@@H](CS)C(O)=N[C@@H](C)C(O)=N[C@@H](CO)C(O)=NCC(O)=N[C@@H](CCC(=N)O)C(O)=NC(CS)C(O)=N[C@@H](C(O)=N[C@H](CS)C(O)=N[C@H](CO)C(O)=NCC(O)=N[C@H](CS)C(O)=NCC(O)=N[C@H](CCCCN)C(=O)O)[C@H](C)O)[C@H](C)O']:
                    print(idx_j)
                    pass
                else:
                    smiles_train_valid.append(smi)
                    labels_train_valid.append(labels_train[idx_j])

            data_dict_train = {'0': smiles_train_valid, '1': labels_train_valid}
            dataframe_train = pd.DataFrame(data_dict_train)
            dataframe_train.to_csv(f'./data/{name}-train-{idx}.csv', index=None)

            smiles_test_valid, labels_test_valid = [], []
            smiles_test = smiles_list[mask_test_fold]
            labels_test = labels[mask_test_fold]
            for idx_j, smi in enumerate(smiles_test):
                if smi in [r'[H]/N=C(\N)SCCCc1cnc[nH]1', 'C[C@H](N=C(O)CN=C(O)[C@H](CO)N=C(O)[C@H](CS)N=C(O)CN=C(O)C(CS)N=C(O)C(CC(=O)O)N=C(O)CN)C(O)=N[C@@H](CO)C(O)=N[C@@H](CO)C(O)=N[C@@H](CS)C(O)=N[C@H](C(O)=N[C@@H](CS)C(O)=N[C@@H](C)C(O)=N[C@@H](CO)C(O)=NCC(O)=N[C@@H](CCC(=N)O)C(O)=NC(CS)C(O)=N[C@@H](C(O)=N[C@H](CS)C(O)=N[C@H](CO)C(O)=NCC(O)=N[C@H](CS)C(O)=NCC(O)=N[C@H](CCCCN)C(=O)O)[C@H](C)O)[C@H](C)O']:
                    print(idx_j)
                    pass
                else:
                    smiles_test_valid.append(smi)
                    labels_test_valid.append(labels_test[idx_j])
            data_dict_test = {'0': smiles_test_valid, '1': labels_test_valid}
            dataframe_test = pd.DataFrame(data_dict_test)
            dataframe_test.to_csv(f'./data/{name}-test-{idx}.csv', index=None)
        elif name == '1O2':
            smiles_train_valid, labels_train_valid = [], []
            smiles_train = smiles_list[mask_train_fold]
            labels_train = labels[mask_train_fold]
            for idx_j, smi in enumerate(smiles_train):
                if smi in [r'[H]/N=C(\N)NCCC',]:
                    print(idx_j)
                    pass
                else:
                    smiles_train_valid.append(smi)
                    labels_train_valid.append(labels_train[idx_j])

            data_dict_train = {'0': smiles_train_valid, '1': labels_train_valid}
            dataframe_train = pd.DataFrame(data_dict_train)
            dataframe_train.to_csv(f'./data/{name}-train-{idx}.csv', index=None)

            smiles_test_valid, labels_test_valid = [], []
            smiles_test = smiles_list[mask_test_fold]
            labels_test = labels[mask_test_fold]
            for idx_j, smi in enumerate(smiles_test):
                if smi in [r'[H]/N=C(\N)NCCC', ]:
                    print(idx_j)
                    pass
                else:
                    smiles_test_valid.append(smi)
                    labels_test_valid.append(labels_test[idx_j])
            data_dict_test = {'0': smiles_test_valid, '1': labels_test_valid}
            dataframe_test = pd.DataFrame(data_dict_test)
            dataframe_test.to_csv(f'./data/{name}-test-{idx}.csv', index=None)
        else:
            data_dict_train = {'0': smiles_list[mask_train_fold], '1': labels[mask_train_fold]}
            dataframe_train = pd.DataFrame(data_dict_train)
            dataframe_train.to_csv(f'./data/{name}-train-{idx}.csv', index=None)

            data_dict_test = {'0': smiles_list[mask_test_fold], '1': labels[mask_test_fold]}
            dataframe_test = pd.DataFrame(data_dict_test)
            dataframe_test.to_csv(f'./data/{name}-test-{idx}.csv', index=None)






