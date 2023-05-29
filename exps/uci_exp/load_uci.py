'''
Loads the UCI MFS data and returns the dataloaders for the different views.
'''


import torch
import random
import mvlearn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloaders(batch_size, seed=0):
    random.seed(seed)

    g = torch.Generator()
    g.manual_seed(seed)
    torch.manual_seed(seed)

    views, y = mvlearn.datasets.load_UCImultifeature()

    all_ind = set(np.arange(len(y)))

    unlab_ind = random.sample(
        list(np.arange(len(y))), k=round(len(y) * 0.8))
    lab_ind = [x for x in all_ind if x not in unlab_ind]

    assert len(lab_ind)+len(unlab_ind) == len(
        all_ind), f'data size mismatch... {len(all_ind), len(unlab_ind), len(lab_ind)}'

    # load subset of views to increase local correlations
    unlab_X = [views[i][unlab_ind, :] for i in [0, 4, 5]]
    lab_X = [views[i][lab_ind, :] for i in [0, 4, 5]]
    lab_y = y[lab_ind]

    for i in range(3):
        scaler = StandardScaler()
        scaler.fit(unlab_X[i])
        unlab_X[i] = scaler.transform(unlab_X[i])
        lab_X[i] = scaler.transform(lab_X[i])

    unlab_ds = TensorDataset(*[torch.Tensor(unlab_X[i])
                               for i in range(len(unlab_X))])
    lab_ds = TensorDataset(*[*[torch.Tensor(lab_X[i])
                               for i in range(len(lab_X))], torch.Tensor(lab_y).unsqueeze(1)])

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