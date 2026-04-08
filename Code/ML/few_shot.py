# environment: PhD-RC
import numpy as np
import json
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import xgboost as xgb
import matplotlib.pyplot as plt
import joblib
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


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


def evaluation_cv_few_shot(model, few_shot_size, smiles, embeddings, labels, seed=0, k_fold=5):
    assert embeddings.shape[0] == labels.shape[0]
    assert embeddings.shape[0] == smiles.shape[0]
    mask_cv_fold = k_fold_mask(embeddings.shape[0], seed=seed, k_fold=k_fold)
    res = []
    for idx, mask_test_fold in enumerate(mask_cv_fold):
        mask_train_fold = [_ for _ in range(embeddings.shape[0]) if _ not in mask_test_fold]
        res_few_shot = []
        for idx_few_shot in range(1, 6, 1):
            print(f'fold_index: {idx}, few_shot_idx: {idx_few_shot}')
            model_ori = clone(model)
            train_size = int(idx_few_shot*0.2*len(mask_train_fold))
            train_size = few_shot_size[idx_few_shot-1]
            print(f'The size of this train set is {train_size}')
            np.random.seed(idx_few_shot)
            np.random.shuffle(mask_train_fold)
            mask_train_few_shot = list(np.array(mask_train_fold)[0:train_size])

            x_train, x_test = embeddings[mask_train_few_shot, :], embeddings[mask_test_fold, :]
            y_train, y_test = labels[mask_train_few_shot], labels[mask_test_fold]

            model_ori.fit(x_train, y_train)
            y_train_pred = model_ori.predict(x_train)
            y_test_pred = model_ori.predict(x_test)
            r2_train, r2_test = r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)
            mae_train, mae_test = mean_absolute_error(y_train, y_train_pred), mean_absolute_error(y_test, y_test_pred)
            rmse_train, rmse_test = np.sqrt(mean_squared_error(y_train, y_train_pred)), np.sqrt(mean_squared_error(y_test, y_test_pred))
            metric = [r2_train, mae_train, rmse_train, r2_test, mae_test, rmse_test]
            res_few_shot.append(metric)
        res.append(res_few_shot)
    return res


svr = SVR()
rf = RandomForestRegressor(random_state=42)
name_list = ['OH', 'SO4-', 'O3', '1O2', 'Fe(VI)', 'HClO']

for name in name_list:
    if name in ['1O2', 'OH', 'HClO']:
        fp = 'maccs'
        model = rf
    elif name == 'SO4-':
        fp = 'maccs'
        model = svr
    elif name in ['O3',  'Fe(VI)']:
        fp = 'rdkit'
        model = rf
    if name == 'OH':
        few_shot_size = [400, 800, 1200, 1600, 2000]
    elif name == 'SO4-':
        few_shot_size = [75, 150, 225, 300, 375]
    elif name == 'O3':
        few_shot_size = [100, 200, 300, 400, 500]
    elif name == 'Fe(VI)':
        few_shot_size = [30, 60, 90, 120, 150]
    elif name == 'HClO':
        few_shot_size = [40, 80, 120, 160, 200]
    elif name == '1O2':
        few_shot_size = [25, 50, 75, 100, 125]

    print(f'start to process {name}')
    data = np.loadtxt(f'../../Dataset/{name}.txt', dtype=str, comments=None)
    smiles_list = data[:, 0]
    labels = data[:, 1].astype(float)
    with open(f'../../Dataset/Embeddings/{name}-{fp}.json', "r", encoding="utf-8") as f:
        map_smiles_to_embeddings = json.load(f)
    embeddings = np.array([map_smiles_to_embeddings[smi] for smi in smiles_list])

    res = evaluation_cv_few_shot(model, few_shot_size, smiles_list, embeddings, labels)
    np.save(f'Result/few_shot/{name}.npy', res)

