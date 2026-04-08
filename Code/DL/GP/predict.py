import torch
import numpy as np
from GP import fit_and_predict_gpytorch, load_gpytorch_checkpoint, standardize_y
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import gpytorch


average_dataset = [9.969553987777314, 8.664001611068265, 2.8064615471490826, 7.256930719656284, 1.4439868682191837, 4.069083610368902]
std_dataset= [1.1458978096868415, 1.1755129639568271, 2.5302268427480605, 1.07971901800572, 1.5920809957969, 2.9460550972872412]

name_list = ['OH', 'SO4-', 'O3', '1O2', 'Fe(VI)', 'HClO']
y_pred_list = []
sigma_pred_list = []
for idx_name, name in enumerate(name_list):
    for seed in [0]:
        train_data_dl = torch.load(f"../Result/infos/train-{name}-{seed}.pt")
        test_data_dl = torch.load(f"../Result/infos/test-typical-{name}-{seed}.pt")

        train_smiles, train_labels = train_data_dl['names'], train_data_dl['labels'].astype(float)
        train_emb = np.array(train_data_dl['embeddings']).astype(float)
        test_smiles, test_labels = test_data_dl['names'], test_data_dl['labels'].astype(float)
        test_emb = np.array(test_data_dl['embeddings']).astype(float)

        X_test = torch.as_tensor(test_emb, dtype=torch.float32, device='cuda')
        X_train = torch.as_tensor(train_emb, dtype=torch.float32, device='cuda')
        y_train = torch.as_tensor(train_labels, dtype=torch.float32, device='cuda').view(-1)
        X_test = torch.as_tensor(test_emb, dtype=torch.float32, device='cuda')

        y_train_s, y_mean, y_std = standardize_y(y_train)
        ckpt = f'../Result/gp/model/{name}-{seed}.pt'
        model, likelihood, y_mean, y_std, ckpt_infos = load_gpytorch_checkpoint(ckpt, train_emb, y_train_s)

        # --- 预测：返回 mean/std（在标准化空间）---
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_dist = likelihood(model(X_test))
            mean_s = pred_dist.mean  # 标准化后的均值
            std_s = pred_dist.stddev  # 标准化后的标准差

        y_pred = mean_s * y_std + y_mean
        std = std_s * y_std

        sigma_train_set = np.loadtxt(f'../Result/gp/train/{name}-{seed}.txt', dtype=str, comments=None)[:, -1].astype(float)
        sigma_test_set = np.loadtxt(f'../Result/gp/test/{name}-{seed}.txt', dtype=str, comments=None)[:, -1].astype(float)
        sigma_data = np.r_[sigma_train_set, sigma_test_set]

        y_pred = (y_pred - average_dataset[idx_name])/std_dataset[idx_name]
        # std = (std-np.mean(sigma_data))/np.std(sigma_data)
        y_pred_list.append(y_pred.detach().cpu().numpy().tolist())
        sigma_pred_list.append(std.detach().cpu().numpy().tolist())
np.savetxt(f'../Result/gp/typical-mean.txt', np.array(y_pred_list), fmt='%s')
np.savetxt(f'../Result/gp/typical-std.txt', np.array(sigma_pred_list), fmt='%s')





