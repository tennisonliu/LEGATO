'''
Synthetic simulation experiment
'''

import json
import random
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

def gen_synthetic(setting, num_views=2, corr=0.0, 
                    output_dim=100, num_samples=1000,
                    seed=0):
    ''' 
    Setting = 'local'|'global'
    '''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if setting == 'local':
        mean = np.arange(num_views)
        cov = [[1, corr],[corr, 1]]
        cov = np.kron(np.eye(int(num_views/2), dtype=int), cov)
        latent_codes = np.random.multivariate_normal(mean, cov, num_samples)

    else:
        mean = np.arange(num_views)
        std = np.ones(num_views)
        latent_codes = np.random.normal(loc=mean, scale=std, size=(num_samples, num_views))

        for i in range(0, num_views):
            latent_codes[:, i] = (1-corr)*latent_codes[:, i] + (corr)*latent_codes[:, 0]
    
    synth_X = []
    for i in range(num_views):
        transform = nn.Sequential(
            nn.Linear(1, output_dim, bias=False),
            nn.Tanh()
        )

        gen_features = transform(torch.Tensor(latent_codes[:, [i]])).detach().numpy()
        
        synth_X.append(gen_features)

    synth_y = latent_codes

    return synth_X, synth_y


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_dataloaders(synth_data, num_views, batch_size, seed=0):
    random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    torch.manual_seed(seed)
    
    synth_X, synth_y = synth_data
    
    num_samples = synth_X[0].shape[0]
    unlab_ind = int(0.8*num_samples)

    unlab_X = [synth_X[i][:unlab_ind, :] for i in range(num_views)]
    lab_X = [synth_X[i][unlab_ind:, :] for i in range(num_views)]
    lab_y = synth_y[unlab_ind:, :]
    
    unlab_ds = TensorDataset(*[torch.Tensor(unlab_X[i]) for i in range(num_views)])
    lab_ds = TensorDataset(*[*[torch.Tensor(lab_X[i]) for i in range(num_views)], torch.Tensor(lab_y)])

    train_len = int(0.8*len(unlab_ds))
    val_len = len(unlab_ds) - train_len
    train_unlab, val_unlab = random_split(unlab_ds, [train_len, val_len], generator=g)

    train_len = int(0.5*len(lab_ds))
    val_len = int(0.1*len(lab_ds))
    test_len = len(lab_ds) - train_len - val_len
    train_lab, val_lab, test_lab = random_split(lab_ds, [train_len, val_len, test_len], generator=g)

    data_loaders = {
        'train_unlab': DataLoader(train_unlab, batch_size=batch_size, shuffle=True, num_workers=0, worker_init_fn=seed_worker, generator=g),
        'val_unlab': DataLoader(val_unlab, batch_size=batch_size, shuffle=False, num_workers=0, worker_init_fn=seed_worker, generator=g),
        'train_lab': DataLoader(train_lab, batch_size=batch_size, shuffle=True, num_workers=0, worker_init_fn=seed_worker, generator=g),
        'val_lab': DataLoader(val_lab, batch_size=batch_size, shuffle=False, num_workers=0, worker_init_fn=seed_worker, generator=g),
        'test_lab': DataLoader(test_lab, batch_size=batch_size, shuffle=False, num_workers=0, worker_init_fn=seed_worker, generator=g)
    }
    
    return data_loaders

def save_results(pretrain_loss, ft_loss, test_loss, settings, fname):
    res = {}
    for i, loss in enumerate(pretrain_loss):
        num = settings[i]
        res[num] = {}
        res[num]['pretrain'] = loss
        res[num]['ft'] = ft_loss[i]
        res[num]['evaluate'] = test_loss[i]
    res = json.dumps(res)
    f = open(f"{fname}.json", "w")
    f.write(res)
    f.close()