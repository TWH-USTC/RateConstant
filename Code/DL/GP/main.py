import torch
import numpy as np
from GP import fit_and_predict_gpytorch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


name_list = ['OH', 'SO4-', 'O3', '1O2', 'Fe(VI)', 'HClO']
for name in name_list:
    for seed in [0, 1, 2, 3, 4]:
        train_data_dl = torch.load(f"../Result/infos/train-{name}-{seed}.pt")
        test_data_dl = torch.load(f"../Result/infos/test-{name}-{seed}.pt")

        train_smiles, train_labels = train_data_dl['names'], train_data_dl['labels'].astype(float)
        train_emb = np.array(train_data_dl['embeddings']).astype(float)
        test_smiles, test_labels = test_data_dl['names'], test_data_dl['labels'].astype(float)
        test_emb = np.array(test_data_dl['embeddings']).astype(float)

        ckpt = f'../Result/gp/model/{name}-{seed}.pt'
        y_train_pred, sigma_train, y_pred_test, sigma_test, model, likelihood = fit_and_predict_gpytorch(
            train_emb, train_labels, test_emb, iters=300, lr=0.001, save_path=ckpt)

        # print(sigma_train.shape, y_train_pred.shape, train_labels.shape, train_smiles.shape)
        infos_train = np.c_[train_smiles[:, None], train_labels[:, None], y_train_pred[:, None], sigma_train[:, None]]
        infos_test = np.c_[test_smiles[:, None], test_labels[:, None], y_pred_test[:, None], sigma_test[:, None]]

        np.savetxt(f'../Result/gp/train/{name}-{seed}.txt', infos_train, fmt='%s')
        np.savetxt(f'../Result/gp/test/{name}-{seed}.txt', infos_test, fmt='%s')



