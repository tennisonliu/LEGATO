'''
Hyperparameter tuning script
'''

import optuna
from torch import nn
from torch import optim
from training_utils.train_evaluate import pretrain
from method.BaseNetworks import Encoder, Decoder
from method.LEGATO import LEGATO
from training_utils.losses import RegLoss


def hyperopt(input_dims, data_loaders, DEVICE):
    def objective(trial):

        # hyperparameter search range
        hidden_dim = trial.suggest_int('hidden_dim', 40, 100, step=10)
        alpha = trial.suggest_categorical('alpha', [0.01, 0.1, 1])
        beta = trial.suggest_categorical('beta', [0.01, 0.1, 1])
        lr = trial.suggest_categorical('lr', [1e-4, 1e-3, 1e-2])
        weight_decay = trial.suggest_categorical(
            'weight_decay', [1e-4, 1e-3, 1e-2])

        encoder_list = nn.ModuleList(
            Encoder(input_dims[i], hidden_dim, hidden_dim, 1) for i in range(len(input_dims)))
        decoder_list = nn.ModuleList(
            Decoder(hidden_dim, input_dims[i], hidden_dim, 1) for i in range(len(input_dims)))

        criterion = RegLoss(alpha=alpha, beta=beta)

        model = LEGATO(n_views=len(input_dims),
                       n_in_feats=hidden_dim,
                       encoder_list=encoder_list,
                       decoder_list=decoder_list,
                       pool_ratio=0.5,
                       sparse_threshold=0.1,
                       device=DEVICE).to(DEVICE)

        optimiser = optim.Adam(model.parameters(), lr=lr,
                               weight_decay=weight_decay)

        val_loss = pretrain(model, criterion, optimiser, data_loaders,
                            EPOCHS=1000, PATIENCE=20, save_dir=None, seed=0, DEVICE=DEVICE, silent=True)

        return val_loss

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)

    # Print the best hyperparameters and the corresponding loss
    best_params = study.best_params
    best_loss = study.best_value
    print(f'best hyperparameters: {best_params}')
    print(f'best loss: {best_loss:.4f}')

    return best_params
