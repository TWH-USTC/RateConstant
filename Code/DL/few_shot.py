# environment: unimol
from utils import get_data
from data import StandardScaler
import numpy as np
from main import parse_args, train, get_smi_to_embedding_map
from data import MoleculeDataset


name_list = ['OH', 'SO4-', 'O3', '1O2', 'Fe(VI)', 'HClO']  #
for name in name_list:
    print(f'start to process {name}')
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

    for seed in [0, 1, 2, 3, 4]:
        for idx_few_shot in range(5):
            args = parse_args()
            args.atom_features_type = 'unimolv2'
            args.step = 'functional_concat'

            num_epochs = 50
            checkpoint_path = None

            train_data_path = f'./data/{name}-train-{seed}.csv'
            test_data_path = f'./data/{name}-test-{seed}.csv'
            train_dataset = get_data(path=train_data_path, args=args)
            test_dataset = get_data(path=test_data_path, args=args)

            train_dataset.shuffle(seed=idx_few_shot)
            train_dataset = MoleculeDataset(train_dataset[0:few_shot_size[idx_few_shot]])

            if len(train_dataset) < 200:
                batch_size = 4
            elif 200 <= len(train_dataset) < 500:
                batch_size = 8
            elif 500 <= len(train_dataset) < 1000:
                batch_size = 16
            else:
                batch_size = 32
            print(f'The size of this training set is {len(train_dataset)}')

            train_smiles, train_targets = train_dataset.smiles(), train_dataset.targets()
            scaler = StandardScaler().fit(train_targets)
            scaled_targets = scaler.transform(train_targets).tolist()
            train_dataset.set_targets(scaled_targets)
            scaled_targets_test = scaler.transform(test_dataset.targets()).tolist()
            test_dataset.set_targets(scaled_targets_test)

            if args.atom_features_type == 'base':
                func_features = None
            else:
                mappins_smiles_to_embedding = get_smi_to_embedding_map(name, args.atom_features_type)
                emb = []
                for smi in train_dataset.smiles():
                    emb.append(mappins_smiles_to_embedding[smi])
                emb = np.concatenate(emb, axis=0)
                average, std = np.mean(emb, axis=0), np.std(emb, axis=0)
                # func_smi_to_repr = lambda smi: mappins_smiles_to_embedding[smi]
                func_smi_to_repr = lambda smi: (mappins_smiles_to_embedding[smi] - average[None, :])/std[None, :]
                func_features = func_smi_to_repr

            smiles_train, smiles_test, train_labels, train_preds, test_labels, test_preds = \
                train(args, train_dataset, test_dataset, scaler, num_epochs, batch_size, func_features, checkpoint_path)
            train_res = np.array([smiles_train, train_labels, train_preds]).T
            test_res = np.array([smiles_test, test_labels, test_preds]).T
            np.savetxt(f'Result/few_shot/train/{name}-{seed}-{idx_few_shot}-{args.atom_features_type}-{args.step}.txt', train_res, fmt='%s')
            np.savetxt(f'Result/few_shot/test/{name}-{seed}-{idx_few_shot}-{args.atom_features_type}-{args.step}.txt', test_res, fmt='%s')


name_list = ['OH', 'SO4-', 'O3', '1O2', 'Fe(VI)', 'HClO']  #
for name in name_list:
    print(f'start to process {name}')
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

    for seed in [0, 1, 2, 3, 4]:
        for idx_few_shot in range(5):
            args = parse_args()
            args.atom_features_type = 'base'
            args.step = 'none'

            num_epochs = 50
            checkpoint_path = None

            train_data_path = f'./data/{name}-train-{seed}.csv'
            test_data_path = f'./data/{name}-test-{seed}.csv'
            train_dataset = get_data(path=train_data_path, args=args)
            test_dataset = get_data(path=test_data_path, args=args)

            train_dataset.shuffle(seed=idx_few_shot)
            train_dataset = MoleculeDataset(train_dataset[0:few_shot_size[idx_few_shot]])

            if len(train_dataset) < 200:
                batch_size = 4
            elif 200 <= len(train_dataset) < 500:
                batch_size = 8
            elif 500 <= len(train_dataset) < 1000:
                batch_size = 16
            else:
                batch_size = 32
            print(f'The size of this training set is {len(train_dataset)}')

            train_smiles, train_targets = train_dataset.smiles(), train_dataset.targets()
            scaler = StandardScaler().fit(train_targets)
            scaled_targets = scaler.transform(train_targets).tolist()
            train_dataset.set_targets(scaled_targets)
            scaled_targets_test = scaler.transform(test_dataset.targets()).tolist()
            test_dataset.set_targets(scaled_targets_test)

            if args.atom_features_type == 'base':
                func_features = None
            else:
                mappins_smiles_to_embedding = get_smi_to_embedding_map(name, args.atom_features_type)
                emb = []
                for smi in train_dataset.smiles():
                    emb.append(mappins_smiles_to_embedding[smi])
                emb = np.concatenate(emb, axis=0)
                average, std = np.mean(emb, axis=0), np.std(emb, axis=0)
                # func_smi_to_repr = lambda smi: mappins_smiles_to_embedding[smi]
                func_smi_to_repr = lambda smi: (mappins_smiles_to_embedding[smi] - average[None, :]) / std[None,
                                                                                                       :]
                func_features = func_smi_to_repr

            smiles_train, smiles_test, train_labels, train_preds, test_labels, test_preds = \
                train(args, train_dataset, test_dataset, scaler, num_epochs, batch_size, func_features,
                      checkpoint_path)
            train_res = np.array([smiles_train, train_labels, train_preds]).T
            test_res = np.array([smiles_test, test_labels, test_preds]).T
            np.savetxt(
                f'Result/few_shot/train/{name}-{seed}-{idx_few_shot}-{args.atom_features_type}-{args.step}.txt',
                train_res, fmt='%s')
            np.savetxt(
                f'Result/few_shot/test/{name}-{seed}-{idx_few_shot}-{args.atom_features_type}-{args.step}.txt',
                test_res, fmt='%s')




name_list = ['OH', 'SO4-', 'O3', '1O2', 'Fe(VI)', 'HClO']  #
for name in name_list:
    print(f'start to process {name}')
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

    for seed in [0, 1, 2, 3, 4]:
        for idx_few_shot in range(5):
            args = parse_args()
            args.atom_features_type = 'unimolv2'
            args.step = 'none'

            num_epochs = 50
            checkpoint_path = None

            train_data_path = f'./data/{name}-train-{seed}.csv'
            test_data_path = f'./data/{name}-test-{seed}.csv'
            train_dataset = get_data(path=train_data_path, args=args)
            test_dataset = get_data(path=test_data_path, args=args)

            train_dataset.shuffle(seed=idx_few_shot)
            train_dataset = MoleculeDataset(train_dataset[0:few_shot_size[idx_few_shot]])

            if len(train_dataset) < 200:
                batch_size = 4
            elif 200 <= len(train_dataset) < 500:
                batch_size = 8
            elif 500 <= len(train_dataset) < 1000:
                batch_size = 16
            else:
                batch_size = 32
            print(f'The size of this training set is {len(train_dataset)}')

            train_smiles, train_targets = train_dataset.smiles(), train_dataset.targets()
            scaler = StandardScaler().fit(train_targets)
            scaled_targets = scaler.transform(train_targets).tolist()
            train_dataset.set_targets(scaled_targets)
            scaled_targets_test = scaler.transform(test_dataset.targets()).tolist()
            test_dataset.set_targets(scaled_targets_test)

            if args.atom_features_type == 'base':
                func_features = None
            else:
                mappins_smiles_to_embedding = get_smi_to_embedding_map(name, args.atom_features_type)
                emb = []
                for smi in train_dataset.smiles():
                    emb.append(mappins_smiles_to_embedding[smi])
                emb = np.concatenate(emb, axis=0)
                average, std = np.mean(emb, axis=0), np.std(emb, axis=0)
                # func_smi_to_repr = lambda smi: mappins_smiles_to_embedding[smi]
                func_smi_to_repr = lambda smi: (mappins_smiles_to_embedding[smi] - average[None, :]) / std[None,
                                                                                                       :]
                func_features = func_smi_to_repr

            smiles_train, smiles_test, train_labels, train_preds, test_labels, test_preds = \
                train(args, train_dataset, test_dataset, scaler, num_epochs, batch_size, func_features,
                      checkpoint_path)
            train_res = np.array([smiles_train, train_labels, train_preds]).T
            test_res = np.array([smiles_test, test_labels, test_preds]).T
            np.savetxt(
                f'Result/few_shot/train/{name}-{seed}-{idx_few_shot}-{args.atom_features_type}-{args.step}.txt',
                train_res, fmt='%s')
            np.savetxt(
                f'Result/few_shot/test/{name}-{seed}-{idx_few_shot}-{args.atom_features_type}-{args.step}.txt',
                test_res, fmt='%s')

name_list = ['OH', 'SO4-', 'O3', '1O2', 'Fe(VI)', 'HClO']  #
for name in name_list:
    print(f'start to process {name}')
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

    for seed in [0, 1, 2, 3, 4]:
        for idx_few_shot in range(5):
            args = parse_args()
            args.atom_features_type = 'base'
            args.step = 'functional_concat'

            num_epochs = 50
            checkpoint_path = None

            train_data_path = f'./data/{name}-train-{seed}.csv'
            test_data_path = f'./data/{name}-test-{seed}.csv'
            train_dataset = get_data(path=train_data_path, args=args)
            test_dataset = get_data(path=test_data_path, args=args)

            train_dataset.shuffle(seed=idx_few_shot)
            train_dataset = MoleculeDataset(train_dataset[0:few_shot_size[idx_few_shot]])

            if len(train_dataset) < 200:
                batch_size = 4
            elif 200 <= len(train_dataset) < 500:
                batch_size = 8
            elif 500 <= len(train_dataset) < 1000:
                batch_size = 16
            else:
                batch_size = 32
            print(f'The size of this training set is {len(train_dataset)}')

            train_smiles, train_targets = train_dataset.smiles(), train_dataset.targets()
            scaler = StandardScaler().fit(train_targets)
            scaled_targets = scaler.transform(train_targets).tolist()
            train_dataset.set_targets(scaled_targets)
            scaled_targets_test = scaler.transform(test_dataset.targets()).tolist()
            test_dataset.set_targets(scaled_targets_test)

            if args.atom_features_type == 'base':
                func_features = None
            else:
                mappins_smiles_to_embedding = get_smi_to_embedding_map(name, args.atom_features_type)
                emb = []
                for smi in train_dataset.smiles():
                    emb.append(mappins_smiles_to_embedding[smi])
                emb = np.concatenate(emb, axis=0)
                average, std = np.mean(emb, axis=0), np.std(emb, axis=0)
                # func_smi_to_repr = lambda smi: mappins_smiles_to_embedding[smi]
                func_smi_to_repr = lambda smi: (mappins_smiles_to_embedding[smi] - average[None, :]) / std[None,
                                                                                                       :]
                func_features = func_smi_to_repr

            smiles_train, smiles_test, train_labels, train_preds, test_labels, test_preds = \
                train(args, train_dataset, test_dataset, scaler, num_epochs, batch_size, func_features,
                      checkpoint_path)
            train_res = np.array([smiles_train, train_labels, train_preds]).T
            test_res = np.array([smiles_test, test_labels, test_preds]).T
            np.savetxt(
                f'Result/few_shot/train/{name}-{seed}-{idx_few_shot}-{args.atom_features_type}-{args.step}.txt',
                train_res, fmt='%s')
            np.savetxt(
                f'Result/few_shot/test/{name}-{seed}-{idx_few_shot}-{args.atom_features_type}-{args.step}.txt',
                test_res, fmt='%s')



