import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from sklearn.model_selection import KFold
import os

from datasets import RecDataset
from training import train_model
from evaluate import plot_tr_val_loss, model_basin_performance, model_grid_performance
from unet import UNet

# for reproducibility
torch.manual_seed(555)
np.random.seed(222)
torch.set_default_dtype(torch.float64)

AUGMENTATION = False
BATCH_SIZE = 128
NUM_WORKERS = 0
NUM_GPU = torch.cuda.device_count()
DEVICE = torch.device('cuda:0' if NUM_GPU > 0 else 'cpu')
DEVICE_IDS = list(range(NUM_GPU))

NUM_EPOCHS = 150
INPUT_CHANNEL = 3
OUT_CHANNEL = 1

DEPTH = 5
NUM_FILTER = 4
WEIGHTED = False


def train_fold(tr_idx, val_idx, save_path=None, plot_show=False):
    tr_dataset = RecDataset(idx_list=tr_idx, is_train=True, augmentation_bool=AUGMENTATION)
    val_dataset = RecDataset(idx_list=val_idx, is_train=False)

    tr_loader = DataLoader(tr_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    model_init_dict = {'in_channels': INPUT_CHANNEL, 'n_classes': OUT_CHANNEL, 'depth': DEPTH, 'wf': NUM_FILTER}
    model = UNet(**model_init_dict)

    if len(DEVICE_IDS) > 1:
        model = nn.DataParallel(model, device_ids=DEVICE_IDS)
    model = model.to(DEVICE)
    optim_init_dict = {'lr': 0.01, 'weight_decay': 0.0001}
    optimizer = optim.Adam(model.parameters(), **optim_init_dict)
    scheduler = MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)

    model, tr_losses, val_losses = train_model(NUM_EPOCHS, model, tr_loader, val_loader, optimizer, DEVICE,
                                               scheduler=scheduler, weighted=WEIGHTED)

    plot_tr_val_loss(tr_loss=tr_losses, val_loss=val_losses, save_path=save_path + 'loss.png', plot_show=plot_show)
    cc_b, nse_b, rmse_b = model_basin_performance(model=model, dataset=val_dataset,
                                                  save_path=save_path + 'basin_perf.png', plot_show=plot_show)
    cc_g, nse_g, nrmse_g = model_grid_performance(model=model, dataset=val_dataset,
                                                  save_path=save_path + 'grid_perf.png', plot_show=plot_show)

    if save_path:
        if len(DEVICE_IDS) > 1:
            torch.save(model.module.state_dict(), save_path + 'model.pt')
        else:
            torch.save(model.state_dict(), save_path + 'model.pt')

        np.savez(save_path + 'metrics.npz', cc_b=cc_b, nse_b=nse_b,
                 rmse_b=rmse_b, cc_g=cc_g, nse_g=nse_g, nrmse_g=nrmse_g)

        np.savez(save_path + 'loss.npz', tr_loss=tr_losses, val_loss=val_losses)

    return cc_b, nse_b, rmse_b, cc_g, nse_g, nrmse_g


def cross_validating(data_len=226, num_folds=10, shuffle=False):
    splits = KFold(n_splits=num_folds, shuffle=shuffle)
    cc_list, nse_list, rmse_list = [], [], []
    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(data_len))):
        path = str('cross_validation/' + f'{fold}/')
        if not os.path.exists(path):
            os.makedirs(path)

        train_idx, val_idx = list(train_idx), list(val_idx)

        print(f'{fold} fold -----------------------------')

        cc_b, nse_b, rmse_b, *others = train_fold(train_idx, val_idx, save_path=path, plot_show=False)
        cc_list.append(cc_b)
        nse_list.append(nse_b)
        rmse_list.append(rmse_b)
        print()
        print('------------------------------------------')

    cc_a = np.array(cc_list)
    nse_a = np.array(nse_list)
    rmse_a = np.array(rmse_list)
    print(f'ensemble performance:')
    print(f'CC: {np.mean(cc_a):.2f}; NSE: {np.mean(nse_a):.2f}; RMSE: {np.mean(rmse_a):.2f}')

    return cc_a, nse_a, rmse_a


if __name__ == '__main__':
    print('starting...')

    DATA_LEN = 226  # 226
    cross_validating(shuffle=True)
