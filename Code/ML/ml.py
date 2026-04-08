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


def evaluation_cv(model, smiles, embeddings, labels, seed=0, k_fold=5):
    assert embeddings.shape[0] == labels.shape[0]
    assert embeddings.shape[0] == smiles.shape[0]
    mask_cv_fold = k_fold_mask(embeddings.shape[0], seed=seed, k_fold=k_fold)
    train_res = []
    test_res = []
    model_res = []

    for idx, mask_test_fold in enumerate(mask_cv_fold):
        model_ori = clone(model)
        mask_train_fold = [_ for _ in range(embeddings.shape[0]) if _ not in mask_test_fold]
        x_train, x_test = embeddings[mask_train_fold, :], embeddings[mask_test_fold, :]
        y_train, y_test = labels[mask_train_fold], labels[mask_test_fold]
        smiles_train, smiles_test = smiles[mask_train_fold], smiles[mask_test_fold]
        model_ori.fit(x_train, y_train)
        y_train_pred = model_ori.predict(x_train)
        y_test_pred = model_ori.predict(x_test)
        plt.scatter(y_train, y_train_pred)
        plt.scatter(y_test, y_test_pred)
        plt.show()
        model_res.append(model_ori)
        train_res.append(np.concatenate([smiles_train[:, None], y_train[:, None], y_train_pred[:, None]], axis=1))
        test_res.append(np.concatenate([smiles_test[:, None], y_test[:, None], y_test_pred[:, None]], axis=1))
    train_res = np.concatenate(train_res, axis=1)
    test_res = np.concatenate(test_res, axis=1)
    return model_res, train_res, test_res


mlp = MLPRegressor(random_state=42, max_iter=200)
knn = KNeighborsRegressor()
bag = BaggingRegressor(random_state=42)
svr = SVR()
kr = KernelRidge()
dt = DecisionTreeRegressor(random_state=42)
ada = AdaBoostRegressor(random_state=42)
rf = RandomForestRegressor(random_state=42)
et = ExtraTreesRegressor(random_state=42)
xgbr = xgb.sklearn.XGBRegressor(random_state=0, seed=0)
lr = LinearRegression()
ridge = Ridge(random_state=42)
lasso = Lasso(random_state=42)

fps_type = ['rdkit', 'morgan', 'maccs']
models = [xgbr, svr, kr, knn, bag, dt, mlp, rf, et, ada]  #
models_text = ['xgb', 'svr', 'kr', 'knn', 'bag', 'dt', 'mlp', 'rf', 'et', 'ada']
name_list = ['OH', 'SO4-', 'O3', '1O2', 'Fe(VI)', 'HClO']  #

for name in name_list:
    for fp in fps_type:
        data = np.loadtxt(f'../../Dataset/{name}.txt', dtype=str, comments=None)
        smiles_list = data[:, 0]
        labels = data[:, 1].astype(float)
        with open(f'../../Dataset/Embeddings/{name}-{fp}.json', "r", encoding="utf-8") as f:
            map_smiles_to_embeddings = json.load(f)
        embeddings = np.array([map_smiles_to_embeddings[smi] for smi in smiles_list])
        for idx, model in enumerate(models):
            print(f'process {name}-{fp}-{models_text[idx]}')
            model_res, train_res, test_res = evaluation_cv(model, smiles_list, embeddings, labels)
            np.savetxt(f'Result/train/{name}-{fp}-{models_text[idx]}.txt', train_res, fmt='%s')
            np.savetxt(f'Result/test/{name}-{fp}-{models_text[idx]}.txt', test_res, fmt='%s')
            for model_idx in range(len(model_res)):
                # print(f'../../Result/models/{name}-{fp}-{models_text[idx]}-{model_idx}.joblib')
                joblib.dump(model_res[model_idx], f'Result/models/{name}-{fp}-{models_text[idx]}-{model_idx}.joblib')
