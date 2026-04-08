# environment: unimol
from model import initialize_weights, MoleculeModel
from utils import get_data, add_functional_prompt, save_checkpoint
from data import MoleculeDataset, StandardScaler


from argparse import ArgumentParser
import torch
from torch import nn
import h5py
import math

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def get_learning_rate(training_step, num_warmup_steps=1000, num_total_training_steps=20000, max_learning_rate=3e-4, min_learning_rate=3e-5):
    if training_step <= num_warmup_steps:
        lr = (max_learning_rate * training_step) / num_warmup_steps
    elif training_step > num_total_training_steps:
        lr=min_learning_rate
    else:
        ratio_total_steps_post_warmup = (training_step - num_warmup_steps) / (num_total_training_steps - num_warmup_steps)
        cosine_scaler = 0.5 * (1.0 + math.cos(math.pi * ratio_total_steps_post_warmup))
        lr = min_learning_rate + cosine_scaler * (max_learning_rate - min_learning_rate)
    return lr


def learning_rate_scheduler(num_warmup_steps=1000, num_total_training_steps=20000, max_learning_rate=3e-4, min_learning_rate=3e-5):
    def get_lr(training_step):
        return get_learning_rate(training_step, num_warmup_steps, num_total_training_steps, max_learning_rate, min_learning_rate)
    return get_lr


def update_lr_optimizer(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_smi_to_embedding_map(dataset_name, model_name='unimolv1'):
    path = '../../Dataset/Embeddings/' + dataset_name + '-' + model_name + '.h5'
    with h5py.File(path, "r") as f:
        names = [x.decode("utf-8") for x in f["names"][...]]
        g = f["mats"]
        mats = [g[f"{i:06d}_{name}"][...] for i, name in enumerate(names)]
    return dict(zip(names, mats))


def load_checkpoint(path):
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

    return model


def update_lr_epoch(optimizer, epoch, total_epochs=100, base_lr=1e-3, min_lr=1e-5, warmup_epochs=10):
    if total_epochs <= 0:
        raise ValueError("total_epochs must be > 0")
    epoch = max(0, min(epoch, total_epochs))

    if warmup_epochs > 0 and epoch < warmup_epochs:
        lr = base_lr * (epoch / float(warmup_epochs))
    else:
        t = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        t = min(max(t, 0.0), 1.0)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))

    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset_type', type=str, default='regression')
    parser.add_argument('--multiclass_num_classes', type=str, default=None)
    parser.add_argument('--features_only', type=str, default=False)
    parser.add_argument('--features_size', type=str, default=None)
    parser.add_argument('--use_input_features', type=str, default=None)  # classification
    parser.add_argument('--num_tasks', type=int, default=1)  # classification
    parser.add_argument('--encoder_name', type=str, default='CMPNN')

    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--activation', type=str, default='ReLU')
    parser.add_argument('--step', type=str, default='functional_concat')  #
    parser.add_argument("--output_size", type=int, default=1)
    parser.add_argument('--ffn_num_layers', type=int, default=2)
    parser.add_argument('--ffn_hidden_size', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=300)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--atom_messages', type=bool, default=False)
    parser.add_argument('--bias', type=bool, default=False)
    parser.add_argument('--undirected', type=bool, default=False)

    parser.add_argument('--features_path', type=str, default=None)
    parser.add_argument('--max_data_size', type=int, default=None)
    parser.add_argument('--use_compound_names', type=bool, default=False)
    parser.add_argument('--features_generator', type=str, default=None)

    parser.add_argument('--atom_features_type', type=str, default='base')  #

    parser.add_argument('--cuda', type=bool, default=True)
    args = parser.parse_args()
    return args


def evaluate(model, dataset, batch_size=256):
    model.eval()
    preds_all = []
    labels_all = []
    smiles_all = []

    for st in range(0, len(dataset), batch_size):
        ed = min(len(dataset), st + batch_size)
        mol_batch = MoleculeDataset(dataset[st:ed])
        smiles_batch, features_batch, target_batch = mol_batch.smiles(), mol_batch.features(), mol_batch.targets()

        prompt = False
        step = 'finetune'
        with torch.no_grad():
            preds = model(step, prompt, smiles_batch, features_batch).detach().cpu().numpy()
        smiles_all.append(smiles_batch)
        labels_all.append(target_batch)
        preds_all.append(preds)
    return np.concatenate(smiles_all, axis=0), np.concatenate(preds_all, axis=0)[:, 0], np.concatenate(labels_all, axis=0)[:, 0]


def train(args, train_dataset, test_dataset, scaler, num_epochs, batch_size, func_features, checkpoint_path=None):
    model = MoleculeModel(classification=False, multiclass=False, pretrain=False)
    model.create_encoder(args, encoder_name=args.encoder_name, func=func_features)
    model.create_ffn(args)
    initialize_weights(model)
    if args.step == 'functional_prompt':
        add_functional_prompt(model, args)
    model = model.cuda()

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        train_dataset.shuffle(seed=epoch)
        model.train()
        for st in range(0, len(train_dataset), batch_size):
            ed = min(len(train_dataset), st + batch_size)
            mol_batch = MoleculeDataset(train_dataset[st:ed])
            smiles_batch, features_batch, target_batch = mol_batch.smiles(), mol_batch.features(), mol_batch.targets()
            target_batch = torch.from_numpy(np.array(target_batch)).cuda().to(torch.float32)
            prompt = False
            step = 'finetune'
            preds = model(step, prompt, smiles_batch, features_batch).to(torch.float32)
            loss = criterion(preds, target_batch)
            optimizer.zero_grad(set_to_none=True)
            update_lr_epoch(optimizer, epoch)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            loss.backward()
            optimizer.step()

        _, train_preds, train_labels = evaluate(model, train_dataset)
        _, test_preds, test_labels = evaluate(model, test_dataset)
        train_preds = scaler.inverse_transform(train_preds)
        train_labels = scaler.inverse_transform(train_labels)
        test_preds = scaler.inverse_transform(test_preds)
        test_labels = scaler.inverse_transform(test_labels)
        rmse_train = np.sqrt(mean_squared_error(train_labels, train_preds))
        rmse_test = np.sqrt(mean_squared_error(test_labels, test_preds))
        mae_train = mean_absolute_error(train_labels, train_preds)
        mae_test = mean_absolute_error(test_labels, test_preds)
        r2_train = r2_score(train_labels, train_preds)
        r2_test = r2_score(test_labels, test_preds)

        print(
            f"Step {epoch:02d} | "
            f"[Train] RMSE={rmse_train:.4f}, MAE={mae_train:.4f}, "
            f"R2={r2_train:.4f} | "
            f"[test] RMSE={rmse_test:.4f}, MAE={mae_test:.4f}, "
            f"R2={r2_test:.4f}"
        )
    if checkpoint_path is not None:
        save_checkpoint(checkpoint_path, model=model, args=args)

    smiles_test, test_preds, test_labels = evaluate(model, test_dataset)
    smiles_train, train_preds, train_labels = evaluate(model, train_dataset)
    train_preds = scaler.inverse_transform(train_preds)
    train_labels = scaler.inverse_transform(train_labels)
    test_preds = scaler.inverse_transform(test_preds)
    test_labels = scaler.inverse_transform(test_labels)

    return smiles_train, smiles_test, train_labels, train_preds, test_labels, test_preds


if __name__ == '__main__':
    name_list = ['OH', 'SO4-', 'O3', '1O2', 'Fe(VI)', 'HClO']
    for name in name_list:
        for seed in [0, 1, 2, 3, 4]:
            args = parse_args()
            if name == 'OH':
                batch_size = 32
            elif name in ['SO4-', 'O3',]:
                batch_size = 8
            elif name in ['Fe(VI)', 'HClO', '1O2']:
                batch_size = 4

            num_epochs = 50
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
                # func_smi_to_repr = lambda smi: mappins_smiles_to_embedding[smi]
                func_smi_to_repr = lambda smi: (mappins_smiles_to_embedding[smi] - average[None, :])/std[None, :]
                func_features = func_smi_to_repr

            smiles_train, smiles_test, train_labels, train_preds, test_labels, test_preds = \
                train(args, train_dataset, test_dataset, scaler, num_epochs, batch_size, func_features, checkpoint_path)
            train_res = np.array([smiles_train, train_labels, train_preds]).T
            test_res = np.array([smiles_test, test_labels, test_preds]).T
            np.savetxt(f'Result/train/{name}-{seed}-{args.atom_features_type}-{args.step}.txt', train_res, fmt='%s')
            np.savetxt(f'Result/test/{name}-{seed}-{args.atom_features_type}-{args.step}.txt', test_res, fmt='%s')