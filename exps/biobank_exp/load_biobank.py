'''
Loads the UK Biobank data and returns the dataloaders for the different views.
'''
import os
import json
import torch
import random
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def encode_and_bind(original_dataframe, features_to_encode):
    cont_feats = [col for col in list(
        original_dataframe.columns) if col not in features_to_encode]
    dummies = pd.get_dummies(original_dataframe[features_to_encode])
    res = pd.concat([original_dataframe[cont_feats], dummies], axis=1)
    return (res)


def read_biobank(samples=2000, seed=0):
    data = pd.read_feather(
        '../../data/biobank_edited_all_participants.feather')
    # drop all columns with more than 25% missingness
    data = data.dropna(thresh=data.shape[0]*0.75, how='all', axis=1)
    # drop all rows with 1% missingness
    data = data.dropna(thresh=data.shape[1]*0.99, axis=0)
    data = data.reset_index()

    random_index = data.sample(samples, random_state=seed).index
    pos_index = data['death_lca_primary'].loc[lambda x: x == 1].index
    indices = random_index.append(pos_index)

    data_subset = data.iloc[indices, :]

    views = []
    for fname in os.listdir():
        if fname.endswith('.json') and fname not in ['label.json', 'configs.json']:
            with open(fname, 'r') as f:
                feats = json.load(f)
                views.append(data_subset[list(set(feats))])

    encoded_views = []
    for view in views:

        discard_cols = list(view.select_dtypes(include=['datetime64']).columns)
        view = view.drop(discard_cols, axis=1)

        cat_cols = list(view.select_dtypes(
            include=['object', 'category']).columns)
        encoded_views.append(encode_and_bind(view, cat_cols))

    imputed_views = []
    for view in encoded_views:
        view = view.fillna(view.mean())
        imputed_views.append(view.values)

    with open('label.json', 'r') as f:
        feats = json.load(f)
        y = data_subset[feats].values

    return imputed_views, y


def get_dataloaders(batch_size, samples=5000, seed=0):
    random.seed(seed)

    g = torch.Generator()
    g.manual_seed(seed)
    torch.manual_seed(seed)

    views, y = read_biobank(samples=samples, seed=seed)
    num_views = len(views)

    all_ind = set(np.arange(len(y)))

    unlab_ind = random.sample(
        list(np.arange(len(y))), k=round(len(y) * 0.8))
    lab_ind = [x for x in all_ind if x not in unlab_ind]

    assert len(lab_ind)+len(unlab_ind) == len(
        all_ind), f'data size mismatch... {len(all_ind), len(unlab_ind), len(lab_ind)}'

    unlab_X = [views[i][unlab_ind, :] for i in range(num_views)]
    lab_X = [views[i][lab_ind, :] for i in range(num_views)]
    lab_y = y[lab_ind]

    for i in range(num_views):
        scaler = StandardScaler()
        scaler.fit(unlab_X[i])
        unlab_X[i] = scaler.transform(unlab_X[i])
        lab_X[i] = scaler.transform(lab_X[i])

    unlab_ds = TensorDataset(*[torch.Tensor(unlab_X[i])
                               for i in range(len(unlab_X))])
    lab_ds = TensorDataset(*[*[torch.Tensor(lab_X[i])
                               for i in range(len(lab_X))], torch.Tensor(lab_y)])

    train_len = int(0.8*len(unlab_ds))
    val_len = len(unlab_ds) - train_len
    train_unlab, val_unlab = random_split(
        unlab_ds, [train_len, val_len], generator=g)

    train_len = int(0.6*len(lab_ds))
    val_len = int(0.2*len(lab_ds))
    test_len = len(lab_ds) - train_len - val_len
    train_lab, val_lab, test_lab = random_split(
        lab_ds, [train_len, val_len, test_len], generator=g)

    data_loaders = {
        'train_unlab': DataLoader(train_unlab, batch_size=batch_size, shuffle=True, num_workers=0, worker_init_fn=seed_worker, generator=g),
        'val_unlab': DataLoader(val_unlab, batch_size=batch_size, shuffle=False, num_workers=0, worker_init_fn=seed_worker, generator=g),
        'train_lab': DataLoader(train_lab, batch_size=batch_size, shuffle=True, num_workers=0, worker_init_fn=seed_worker, generator=g),
        'val_lab': DataLoader(val_lab, batch_size=batch_size, shuffle=False, num_workers=0, worker_init_fn=seed_worker, generator=g),
        'test_lab': DataLoader(test_lab, batch_size=batch_size, shuffle=False, num_workers=0, worker_init_fn=seed_worker, generator=g)
    }

    return data_loaders
