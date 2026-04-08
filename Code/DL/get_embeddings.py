from model import initialize_weights, MoleculeModel
from data import MoleculeDataset, StandardScaler
import torch
import numpy as np
from utils import get_data, add_functional_prompt
from main import parse_args, get_smi_to_embedding_map, evaluate


def get_embeddings(model, data, batch_size=64):
    smiles_all = []
    embeddings_all = []
    labels_all = []
    preds_all = []
    for st in range(0, len(data), batch_size):
        ed = min(len(data), st + batch_size)
        mol_batch = MoleculeDataset(data[st:ed])
        smiles_batch, features_batch, target_batch = mol_batch.smiles(), mol_batch.features(), mol_batch.targets()

        batch = smiles_batch
        prompt = False
        step = 'finetune'

        with torch.no_grad():
            embeddings = model.encoder(step, prompt, batch, features_batch).detach().cpu().numpy()
            preds = model(step, prompt, batch, features_batch).detach().cpu().numpy()
        smiles_all.append(smiles_batch)
        embeddings_all.append(embeddings)
        labels_all.append(target_batch)
        preds_all.append(preds)

    smiles = np.concatenate(smiles_all, axis=0)
    embeddings = np.concatenate(embeddings_all, axis=0)
    labels = np.concatenate(labels_all, axis=0)[:, 0]
    preds = np.concatenate(preds_all, axis=0)[:, 0]
    return smiles, embeddings, labels, preds


def load_checkpoint(path, func_features):
    # path = f'./model_trained/{name}-{seed}-{args.atom_features_type}-{args.step}'
    state = torch.load(path, map_location=lambda storage, loc: storage)
    args, loaded_state_dict = state['args'], state['state_dict']
    # build model
    model = MoleculeModel(classification=args.dataset_type == 'classification',
                          multiclass=args.dataset_type == 'multiclass', pretrain=False)

    model.create_encoder(args, args.encoder_name, func=func_features)
    model.create_ffn(args)
    initialize_weights(model)
    # Build model
    if args.step == 'functional_prompt':
        add_functional_prompt(model, args)
    model_state_dict = model.state_dict()

    # Skip missing parameters and parameters of mismatched size
    pretrained_state_dict = {}
    for param_name in loaded_state_dict.keys():
        if param_name not in model_state_dict:
            print(f'Pretrained parameter "{param_name}" cannot be found in model parameters.')
        elif model_state_dict[param_name].shape != loaded_state_dict[param_name].shape:
            print(f'Pretrained parameter "{param_name}" '
                  f'of shape {loaded_state_dict[param_name].shape} does not match corresponding '
                  f'model parameter of shape {model_state_dict[param_name].shape}.')
        else:
            # debug(f'Loading pretrained parameter "{param_name}".')
            pretrained_state_dict[param_name] = loaded_state_dict[param_name]
    # Load pretrained weights
    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(model_state_dict)
    if args.cuda:
        model = model.cuda()
    model.eval()
    return model


if __name__ == '__main__':
    name_list = ['OH', 'SO4-', 'O3', '1O2', 'Fe(VI)', 'HClO']  #
    for name in name_list:
        for seed in [0, 1, 2, 3, 4]:
            print(f"start to process {name}-{seed}")
            args = parse_args()
            args.atom_features_type = 'unimolv2'
            args.step = 'functional_concat'

            checkpoint_path = f'./model_trained/{name}-{seed}-{args.atom_features_type}-{args.step}'

            train_data_path = f'./data/{name}-train-{seed}.csv'
            test_data_path = f'./data/{name}-test-{seed}.csv'
            train_dataset = get_data(path=train_data_path, args=args)
            test_dataset = get_data(path=test_data_path, args=args)

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
                func_smi_to_repr = lambda smi: (mappins_smiles_to_embedding[smi] - average[None, :])/std[None, :]
                func_features = func_smi_to_repr

            model = load_checkpoint(checkpoint_path, func_features=func_features)

            train_smiles, train_embeddings, train_labels, train_preds = get_embeddings(model, train_dataset, 128)
            train_preds = scaler.inverse_transform(train_preds)
            train_labels = scaler.inverse_transform(train_labels)
            test_smiles, test_embeddings, test_labels, test_preds = get_embeddings(model, test_dataset, 128)
            test_preds = scaler.inverse_transform(test_preds)
            test_labels = scaler.inverse_transform(test_labels)

            # mask = np.argsort(np.abs(test_labels - test_preds))
            # print(test_smiles[mask])
            # print(test_labels[mask])
            # print(test_preds[mask])
            #
            # import matplotlib.pyplot as plt
            # from sklearn.metrics import r2_score
            # plt.scatter(test_labels, test_preds)
            # plt.show()
            # print(r2_score(test_labels, test_preds, ))
            # raise ValueError

            train_data = {
                "names": train_smiles,
                "labels": train_labels,
                "preds": train_preds,
                "embeddings": list(train_embeddings),
            }
            torch.save(train_data, f"Result/infos/train-{name}-{seed}.pt")

            test_data = {
                "names": test_smiles,
                "labels": test_labels,
                "preds": test_preds,
                "embeddings": list(test_embeddings),
            }
            torch.save(test_data, f"Result/infos/test-{name}-{seed}.pt")