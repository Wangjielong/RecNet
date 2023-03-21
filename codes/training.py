# this python script trains our models without the auxiliary network or weighted loss function
import time
import copy

import numpy as np
import torch
import torch.nn as nn

mask_data = np.load('data/mask32x80_a.npz', allow_pickle=True)
mask_a = mask_data['mask32x80_a']
MASK_t = torch.from_numpy(mask_a)

down_mask_data = np.load('data/down_mask.npz', allow_pickle=True)
down_mask_a = down_mask_data['down_mask']
DOWN_MASK_t = torch.from_numpy(down_mask_a)


def train_epoch(model, dataloader, loss_fn, optimizer, device, weighted=False):
    running_loss = 0.0
    model.train()
    mask_t = MASK_t.to(device)
    down_mask_t = DOWN_MASK_t.to(device)
    for month_index, _, input_grid, _, target_grid in dataloader:
        input_grid, target_grid = input_grid.to(device), target_grid.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        output = model(input_grid)

        if weighted:
            loss1 = torch.mean(loss_fn(output * mask_t, target_grid * mask_t), dim=(1, 2, 3))
            loss2 = torch.mean(loss_fn(output * down_mask_t, target_grid * down_mask_t), dim=(1, 2, 3))
            loss = loss1.mean() + loss2.mean() * 8
        else:
            loss = loss_fn(output * mask_t, target_grid * mask_t)

        loss.backward()
        optimizer.step()

        # record training loss.
        # loss.item() contains the loss of entire mini-batch, but divided by the batch size
        running_loss += loss.item() * input_grid.size(0)

    return running_loss


def validate_epoch(model, dataloader, loss_fn, device, weighted=False):
    running_loss = 0.0
    model.eval()
    mask_t = MASK_t.to(device)
    down_mask_t = DOWN_MASK_t.to(device)
    with torch.no_grad():
        for month_index, _, input_grid, _, target_grid in dataloader:
            input_grid, target_grid = input_grid.to(device), target_grid.to(device)
            output = model(input_grid)

            if weighted:
                # loss = torch.mean(loss_fn(output * mask_t, target_grid * mask_t), dim=(1, 2, 3))
                # loss = torch.mean(loss * get_seasonal_weight(month_index, device))
                loss1 = torch.mean(loss_fn(output * mask_t, target_grid * mask_t), dim=(1, 2, 3))
                loss2 = torch.mean(loss_fn(output * down_mask_t, target_grid * down_mask_t), dim=(1, 2, 3))
                loss = loss1.mean() + loss2.mean() * 8
            else:
                loss = loss_fn(output * mask_t, target_grid * mask_t)

            running_loss += loss.item() * input_grid.size(0)

    return running_loss


def train_model(num_epochs, model, tr_loader, val_loader, optimizer, device, scheduler=None, weighted=False):
    print('start training and validating...')
    since = time.time()

    if weighted:
        loss_fn = nn.MSELoss(reduction='none')
    else:
        loss_fn = nn.MSELoss()

    tr_losses, val_losses = [], []
    best_model = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)
        tr_loss = train_epoch(model, tr_loader, loss_fn, optimizer, device, weighted=weighted)
        val_loss = validate_epoch(model, val_loader, loss_fn, device, weighted=weighted)

        if scheduler:
            scheduler.step()

        tr_loss = tr_loss / len(tr_loader.dataset)  # loss per epoch
        val_loss = val_loss / len(val_loader.dataset)  # loss per epoch

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())

        tr_losses.append(tr_loss)
        val_losses.append(val_loss)

        print(f'Training loss: {tr_loss:.2f}  Validation loss: {val_loss:.2f}')
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:2f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model)
    return model, tr_losses, val_losses


def get_seasonal_weight(month_index, device):
    """
    the weights increase as the month index increases from 1 to 7, while decreases as that index from 7 to 12
    """
    weight_a = np.sin(month_index.cpu().detach().numpy() * np.pi / 13)
    weight_t = torch.from_numpy(weight_a).view(-1, 1, 1, 1)
    return weight_t.to(device)


if __name__ == '__main__':
    print('start from here')
    # month_index = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    # m_t = torch.from_numpy(month_index)
    # w = get_seasonal_weight(m_t, device='cpu')
