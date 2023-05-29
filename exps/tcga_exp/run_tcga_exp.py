'''
This script runs the tcga experiment for 10 times and saves the results in ../results/tcga_acc_LEGATO.json
'''

import torch
import json
import sys
sys.path.append("../")
sys.path.append("../..")
import numpy as np
from load_tcga import get_dataloaders
from method.LEGATO import LEGATO
from method.BaseNetworks import Encoder, Decoder
from method.LinearProbe import LinearProbe
from training_utils.losses import RegLoss
from training_utils.train_evaluate import pretrain, finetune, evaluate
from training_utils.logging import save_results
from torch import optim
from torch import nn

exp_name = f'tcga_acc'
SEEDS = 10
batch_size = 64
input_dims = [100, 100, 100, 100]

run_pretrain_ls = []
run_finetue_ls = []
run_evaluate_ls = []
run_auroc_ls = []
run_acc_ls = []

# load experimental configs
with open('configs.json', 'r') as f:
    configs = json.load(f)

print(f'running experiments on tcga datasets for {SEEDS} runs')

for SEED in range(SEEDS):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    print(f'processing run {SEED+1}...')

    data_loaders = get_dataloaders(batch_size)

    hidden_dim = configs['hidden_dim']
    lr = configs['lr']
    weight_decay = configs['weight_decay']
    DEVICE = 'cuda'
    PATIENCE = 20
    EPOCHS = 1000

    encoder_list = nn.ModuleList(
        Encoder(input_dims[i], hidden_dim, hidden_dim, 1) for i in range(len(input_dims)))
    decoder_list = nn.ModuleList(
        Decoder(hidden_dim, input_dims[i], hidden_dim, 1) for i in range(len(input_dims)))

    criterion = RegLoss(alpha=configs['alpha'], beta=configs['beta'])

    model = LEGATO(n_views=len(input_dims),
                   n_in_feats=hidden_dim,
                   encoder_list=encoder_list,
                   decoder_list=decoder_list,
                   pool_ratio=0.5,
                   sparse_threshold=0.1,
                   device=DEVICE).to(DEVICE)

    optimiser = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)

    save_dir = f'../local/{exp_name}_{model.name}_pretrained.pt'

    val_loss = pretrain(model, criterion, optimiser, data_loaders,
                        EPOCHS, PATIENCE, save_dir, SEED, DEVICE, silent=True)

    model.load_state_dict(torch.load(save_dir))

    lr = 1e-3
    weight_decay = 1e-3
    EPOCHS = 100
    PATIENCE = 20
    save_dir = f'../local/{exp_name}_{model.name}_downstream.pt'

    downstream_model = LinearProbe(
        encoder=model, n_in_feats=model.latent_dims, n_targets=1, freeze_encoder=True).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.Tensor([5.412087912087912]).to(DEVICE))
    optimiser = optim.Adam(downstream_model.parameters(),
                           lr=lr, weight_decay=weight_decay)

    finetune_loss = finetune(downstream_model, criterion, optimiser,
                             data_loaders, EPOCHS, PATIENCE, save_dir, SEED, DEVICE, silent=True)

    downstream_model.load_state_dict(torch.load(save_dir))
    test_loss, auroc, acc = evaluate(
        downstream_model, criterion, data_loaders, clas=True, DEVICE=DEVICE)

    print(
        f'[run {SEED+1}] test loss: {test_loss:.4f}, auroc: {auroc:.4f}, acc: {acc:.4f}')
    print('\n')

    run_pretrain_ls.append(val_loss)
    run_finetue_ls.append(finetune_loss)
    run_evaluate_ls.append(test_loss)
    run_auroc_ls.append(auroc.item())
    run_acc_ls.append(acc)

print('average results:')
print(f'AUROC: {np.mean(run_auroc_ls):.4f}+/-{np.std(run_auroc_ls):.4f}')
print(f'ACC: {np.mean(run_acc_ls):.4f}+/-{np.std(run_acc_ls):.4f}')

fname = f'../results/{exp_name}_{model.name}'
save_results(run_pretrain_ls, run_finetue_ls, run_evaluate_ls,
             run_auroc_ls, run_acc_ls, fname)
