'''
Training and evaluation functions for pretraining and finetuning.
'''

import torch
import numpy as np
from torch import nn
from torchmetrics import AUROC
from sklearn.metrics import accuracy_score


def pretrain(model, criterion, optimiser, data_loaders, EPOCHS, PATIENCE, save_dir=None, seed=0, DEVICE='cuda', silent=True):
    torch.manual_seed(seed)
    np.random.seed(seed)

    patience = 0
    best_loss = np.inf

    for epoch in range(EPOCHS):

        train_loss = 0
        train_recon_loss = 0
        train_recon_graph_loss = 0
        train_cos_loss = 0
        train_graph_loss = 0
        train_samples = 0

        # train
        for i, data in enumerate(data_loaders['train_unlab']):
            views = [view.to(DEVICE) for view in data]
            model.train()
            optimiser.zero_grad()

            out, latent_graph, input_graph, misc = model(views)

            loss, recon_loss, recon_graph_loss, cos_loss, graph_loss = criterion(
                views, out, latent_graph[0], input_graph, misc[2])
            loss.backward()
            optimiser.step()

            train_loss += loss.item()*views[0].shape[0]
            train_recon_graph_loss += recon_graph_loss.item()*views[0].shape[0]
            train_recon_loss += recon_loss.item()*views[0].shape[0]
            train_cos_loss += cos_loss.item()*views[0].shape[0]
            train_graph_loss += graph_loss.item()*views[0].shape[0]

            train_samples += views[0].shape[0]

        av_train_loss = train_loss/train_samples
        av_train_recon_loss = train_recon_loss/train_samples
        av_train_recon_graph_loss = train_recon_graph_loss/train_samples
        av_train_cos_loss = train_cos_loss/train_samples
        av_train_graph_loss = train_graph_loss/train_samples

        val_loss = 0
        val_recon_loss = 0
        val_recon_graph_loss = 0
        val_cos_loss = 0
        val_graph_loss = 0
        val_samples = 0

        # validation
        with torch.no_grad():
            for i, data in enumerate(data_loaders['val_unlab']):
                views = [view.to(DEVICE) for view in data]
                model.eval()
                out, latent_graph, input_graph, misc = model(views)

                loss, recon_loss, recon_graph_loss, cos_loss, graph_loss = criterion(
                    views, out, latent_graph[0], input_graph, misc[2])

                val_loss += loss.item()*views[0].shape[0]
                val_recon_loss += recon_loss.item()*views[0].shape[0]
                val_recon_graph_loss += recon_graph_loss.item() * \
                    views[0].shape[0]
                val_cos_loss += cos_loss.item()*views[0].shape[0]
                val_graph_loss += graph_loss.item()*views[0].shape[0]

                val_samples += views[0].shape[0]

            av_val_loss = val_loss/val_samples
            av_val_recon_loss = val_recon_loss/val_samples
            av_val_recon_graph_loss = val_recon_graph_loss/val_samples
            av_val_cos_loss = val_cos_loss/val_samples
            av_val_graph_loss = val_graph_loss/val_samples

        if av_val_loss < best_loss:
            best_loss = av_val_loss
            best_train_loss = av_train_loss
            patience = 0
            if save_dir is not None:
                torch.save(model.state_dict(), save_dir)

        else:
            patience += 1
        if patience == PATIENCE:
            print(
                f'[pretraining] terminated after {epoch} epochs...')
            print(
                f'best train loss: {best_train_loss:.4f}, best val loss: {best_loss:.4f}')
            break

        if not silent and epoch % 5 == 0:
            print(f'[epoch {epoch} | patience {patience}] train loss: {av_train_loss:.4f}, recon loss: {av_train_recon_loss:.4f}, graph recon loss: {av_train_recon_graph_loss:.4f}, cos loss: {av_train_cos_loss:.4f}, graph loss: {av_train_graph_loss:.4f}')
            print(f'\tval loss: {av_val_loss:.4f}, recon loss: {av_val_recon_loss:.4f}, graph recon loss: {av_val_recon_graph_loss:.4f}, cos loss: {av_val_cos_loss:.4f}, graph loss: {av_val_graph_loss:.4f}')

    if patience != PATIENCE:
        print(
            f'[pretraining] terminated after {epoch} epochs...')
        print(
            f'best train loss: {best_train_loss:.4f}, best val loss: {best_loss:.4f}')

    return best_loss


def finetune(model, criterion, optimiser, data_loaders, EPOCHS, PATIENCE, save_dir=None, seed=0, DEVICE='cuda', silent=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    patience = 0
    best_loss = np.inf

    for epoch in range(EPOCHS):

        train_loss = 0
        train_samples = 0

        # train
        for i, data in enumerate(data_loaders['train_lab']):
            views = [view.to(DEVICE) for view in data[:-1]]
            y = data[-1].to(DEVICE)

            if isinstance(criterion, nn.CrossEntropyLoss):
                y = y.squeeze().type(torch.long)

            model.train()
            optimiser.zero_grad()

            out = model(views)
            loss = criterion(out, y)
            loss.backward()
            optimiser.step()

            train_loss += loss.item()*views[0].shape[0]
            train_samples += views[0].shape[0]

        av_train_loss = train_loss/train_samples

        val_loss = 0
        val_samples = 0

        # validation
        with torch.no_grad():
            for _, data in enumerate(data_loaders['val_lab']):
                views = [view.to(DEVICE) for view in data[:-1]]
                y = data[-1].to(DEVICE)

                if isinstance(criterion, nn.CrossEntropyLoss):
                    y = y.squeeze().type(torch.long)

                model.eval()
                out = model(views)

                loss = criterion(out, y)

                val_loss += loss.item()*views[0].shape[0]
                val_samples += views[0].shape[0]

            av_val_loss = val_loss/val_samples

        # early stopping
        if av_val_loss < best_loss:
            best_loss = av_val_loss
            best_train_loss = av_train_loss
            patience = 0
            if save_dir is not None:
                torch.save(model.state_dict(), save_dir)
        else:
            patience += 1

        if patience == PATIENCE // 2 and model.freeze_encoder:
            patience = 0
            if not silent:
                print('Unfreezing encoder...')
            for param in model.encoder.parameters():
                param.requires_grad = True
            model.freeze_encoder = False
            optimiser.param_groups[0]['lr'] = 1e-4

        elif patience == PATIENCE:
            print(f'[finetuning] terminated at {epoch} epochs...')
            print(
                f'best train loss: {best_train_loss:.4f}, best val loss: {best_loss:.4f}')
            break

        else:
            pass

        if not silent and epoch % 5 == 0:
            print(
                f'[epoch {epoch}| patience {patience}] train loss: {av_train_loss:.4f}, val loss: {av_val_loss:.4f}')

    if patience != PATIENCE:
        print(f'[finetuning] terminated at {epoch} epochs...')
        print(
            f'best train loss: {best_train_loss:.4f}, best val loss: {best_loss:.4f}')
    return best_loss


def evaluate(model, criterion, data_loaders, clas=False, DEVICE='cuda'):
    test_loss = 0
    test_samples = 0

    if clas:
        preds = []
        target = []

    with torch.no_grad():
        for _, data in enumerate(data_loaders['test_lab']):
            views = [view.to(DEVICE) for view in data[:-1]]
            y = data[-1].to(DEVICE)

            if isinstance(criterion, nn.CrossEntropyLoss):
                y = y.squeeze().type(torch.long)

            model.eval()
            out = model(views)

            loss = criterion(out, y)
            test_loss += loss.item()*views[0].shape[0]
            test_samples += views[0].shape[0]

            if clas:
                preds.append(out.squeeze())
                target.append(y.squeeze())

    av_test_loss = test_loss/test_samples

    if clas:
        preds = torch.cat(preds, dim=0)
        target = torch.cat(target, dim=0).long()

        num_classes = preds.shape[1] if len(preds.shape) == 2 else 1

        if num_classes < 2:
            loss = AUROC(task="binary")
            auroc = loss(preds, target)

            pred_class = (preds > 0.5).long()
            acc = accuracy_score(target.cpu().detach().numpy(),
                                 pred_class.cpu().detach().numpy())
        else:
            assert num_classes > 2
            loss = AUROC(task='multiclass', num_classes=num_classes)
            auroc = loss(preds, target)

            pred_class = torch.softmax(preds, dim=1).argmax(dim=1)
            acc = accuracy_score(target.cpu().detach().numpy(),
                                 pred_class.cpu().detach().numpy())

        return av_test_loss, auroc, acc
    else:
        return av_test_loss
