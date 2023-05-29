'''
Evaluating the impact of number of views on learning.
'''
import sys
sys.path.append("../")
sys.path.append("../..")
import torch
import numpy as np
from torch import optim
from torch import nn
from simulate import gen_synthetic, get_dataloaders, save_results
from method.LEGATO import LEGATO
from method.BaseNetworks import Encoder, Decoder
from method.LinearProbe import LinearProbe
from training_utils.losses import RegLoss
from training_utils.train_evaluate import pretrain, finetune, evaluate
from training_utils.hyperopt import hyperopt

setting = 'global'
exp_name = f'sim_num_views_{setting}'
SEEDS = 5
batch_size = 64
DEVICE = 'cuda'

# simulation setting
corr = 0.5
num_views_ls = [4, 6, 8]
num_samples = 1000
input_dim = 100

pretrain_loss_ls = []
finetune_loss_ls = []
test_loss_ls = []

print(f'investigating effect of K on multi-view learning, with K in {num_views_ls}...')

for num_views in num_views_ls:
    
    run_pretrain_ls = []
    run_finetue_ls = []
    run_test_ls = []
    
    input_dims = [input_dim]*num_views
    
    print(f'tuning hyperparameters for setting with num views: {num_views}')
    
    # tune hyperparameters
    synth_data = gen_synthetic(setting=setting,
                           num_views=num_views, 
                           corr=corr, 
                           num_samples=num_samples, 
                           output_dim=input_dim,
                           seed=0)
    
    data_loaders = get_dataloaders(synth_data, num_views, batch_size)
    best_config = hyperopt(input_dims, data_loaders, DEVICE)
    

    for SEED in range(SEEDS):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        torch.backends.cudnn.deterministic = True

        print(f'processing setting with num views: {num_views}, run {SEED+1}...')

        synth_data = gen_synthetic(setting=setting,
                                   num_views=num_views, 
                                   corr=corr, 
                                   num_samples=num_samples, 
                                   output_dim=input_dim,
                                   seed=0)

        data_loaders = get_dataloaders(synth_data, num_views, batch_size)


        hidden_dim = best_config['hidden_dim']
        lr = best_config['lr']
        weight_decay = best_config['weight_decay']
        PATIENCE = 20
        EPOCHS = 1000

        encoder_list = nn.ModuleList(Encoder(input_dims[i], hidden_dim, hidden_dim, 1) for i in range(num_views))
        decoder_list = nn.ModuleList(Decoder(hidden_dim, input_dims[i], hidden_dim, 1) for i in range(num_views))

        criterion = RegLoss(alpha=best_config['alpha'], beta=best_config['beta'])

        model = LEGATO(n_views=num_views, 
                        n_in_feats=hidden_dim,
                        encoder_list=encoder_list,
                        decoder_list=decoder_list,
                        pool_ratio=0.5,
                        sparse_threshold=0.1,
                        device=DEVICE).to(DEVICE)

        optimiser = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        save_dir = f'../local/{exp_name}_{model.name}_pretrained.pt'

        val_loss = pretrain(model, criterion, optimiser, data_loaders, EPOCHS, PATIENCE, save_dir, 
                              SEED, DEVICE, silent=True)

        model.load_state_dict(torch.load(save_dir))

        lr = 1e-3
        weight_decay = 1e-3
        EPOCHS = 1000
        PATIENCE = 20
        save_dir = f'../local/{exp_name}_{model.name}_downstream.pt'

        downstream_model = LinearProbe(encoder=model, n_in_feats=model.latent_dims, 
                                       n_targets=num_views, freeze_encoder=False).to(DEVICE)

        criterion = nn.MSELoss()
        optimiser = optim.Adam(downstream_model.parameters(), lr=lr, weight_decay=weight_decay)

        finetune_loss = finetune(downstream_model, criterion, optimiser, data_loaders, EPOCHS, PATIENCE, 
                                 save_dir, SEED, DEVICE, silent=True)

        downstream_model.load_state_dict(torch.load(save_dir))
        test_loss = evaluate(downstream_model, criterion, data_loaders, clas=False, DEVICE=DEVICE)

        print(f'[num views {num_views}, run {SEED+1}] test loss: {test_loss:.4f}')
        print('\n')

        run_pretrain_ls.append(val_loss)
        run_finetue_ls.append(finetune_loss)
        run_test_ls.append(test_loss)
        
    pretrain_loss_ls.append(run_pretrain_ls)
    finetune_loss_ls.append(run_finetue_ls)
    test_loss_ls.append(run_test_ls)

for i, perf in enumerate(pretrain_loss_ls):
    print(f'num views: {num_views_ls[i]}')
    print(f'pretraining loss: {np.mean(perf):.4f} +/- {np.std(perf):.4f}')
    print(f'finetune loss: {np.mean(finetune_loss_ls[i]):.4f} +/- {np.std(finetune_loss_ls[i]):.4f}')
    print(f'test loss: {np.mean(test_loss_ls[i]):.4f} +/- {np.std(test_loss_ls[i]):.4f}')

fname = f'../results/{exp_name}_{model.name}'
save_results(pretrain_loss_ls, finetune_loss_ls, test_loss_ls, num_views_ls, fname)