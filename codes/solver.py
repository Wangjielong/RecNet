# this python script trains our models without the auxiliary network or weighted loss function
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import datetime
import os

from datasets import RecDataset, data_length, generate_tr_val_test_idx
from training import train_model

from unet import UNet

# for reproducibility
torch.manual_seed(555)
np.random.seed(222)
torch.set_default_dtype(torch.float64)

DATA_LEN = data_length  # 226
VAL_RATIO, TEST_RATIO = 0.1, 0.1
tr_idx, val_idx, test_idx = generate_tr_val_test_idx(data_len=DATA_LEN, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO,
                                                     shuffle=True)

AUGMENTATION = False
tr_dataset = RecDataset(idx_list=tr_idx, is_train=True, augmentation_bool=AUGMENTATION)
val_dataset = RecDataset(idx_list=val_idx, is_train=False)
test_dataset = RecDataset(idx_list=test_idx, is_train=False)

BATCH_SIZE = 128
NUM_WORKERS = 0
tr_loader = DataLoader(tr_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

NUM_GPU = torch.cuda.device_count()
DEVICE = torch.device('cuda:0' if NUM_GPU > 0 else 'cpu')
DEVICE_IDS = list(range(NUM_GPU))

NUM_EPOCHS = 150
INPUT_CHANNEL = 3
OUT_CHANNEL = 1

DEPTH = 5
NUM_FILTER = 4

model_init_dict = {'in_channels': INPUT_CHANNEL, 'n_classes': OUT_CHANNEL, 'depth': DEPTH, 'wf': NUM_FILTER}
model_name = 'UNet'
model = UNet(**model_init_dict)

if len(DEVICE_IDS) > 1:
    model = nn.DataParallel(model, device_ids=DEVICE_IDS)
model = model.to(DEVICE)

optim_init_dict = {'lr': 0.01, 'weight_decay': 0.0001}
optimizer = optim.Adam(model.parameters(), **optim_init_dict)
scheduler = MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)

WEIGHTED = False
model, tr_losses, val_losses = train_model(NUM_EPOCHS, model, tr_loader, val_loader, optimizer, DEVICE,
                                           scheduler=scheduler, weighted=WEIGHTED)


def record_training_info(save_loss=True, save_model=True, plot_loss=True):
    tr_time = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    path = str('results/' + f'{tr_time}/')
    if not os.path.exists(path):
        os.makedirs(path)

    if save_model:
        np.savez(path + 'tr_val_test_idx.npz', tr_idx=tr_idx, val_idx=val_idx, test_idx=test_idx)
        if len(DEVICE_IDS) > 1:
            torch.save(model.module.state_dict(), path + 'model.pt')
        else:
            torch.save(model.state_dict(), path + 'model.pt')
    if save_loss:
        np.savez(path + 'loss.npz', tr_loss=tr_losses, val_loss=val_losses)

    if plot_loss:
        plot_tr_val_loss(tr_loss=tr_losses, val_loss=val_losses, save_path=path + 'tr_val_loss.png')

    config_file = open(path + 'config.txt', 'a')
    # -----------------------------------------------------------------------------------------
    config_file.write('*' * 80)
    config_file.write('\r\n')
    config_file.write('The training configuration \r\n')
    config_file.write('*' * 80)
    config_file.write('\r\n')
    config_file.write(f'Model: ' + model_name + '\r\n')
    config_file.write(str(model) + '\r\n')
    config_file.write(f'Model initialization: ' + str(model_init_dict) + '\r\n')
    config_file.write(f'Augmentation: {AUGMENTATION}\r\n')
    config_file.write(f'Weighted: {WEIGHTED}\r\n')
    config_file.write(f'the number of the epoch: {NUM_EPOCHS}\r\n')
    config_file.write(f'The percentage of validation (%): {VAL_RATIO * 100}\r\n')
    config_file.write(f'The percentage of testing (%): {TEST_RATIO * 100}\r\n')
    config_file.write(f'Batch size: {BATCH_SIZE}\r\n')
    config_file.write(f'Optimizer: ' + str(optim_init_dict) + '\r\n')
    config_file.write(f'Input channel: {INPUT_CHANNEL}\r\n')
    config_file.write(f'Output channel: {OUT_CHANNEL}\r\n')
    config_file.write(f'The number of GPU: {NUM_GPU}\r\n')
    config_file.close()
    return path


if __name__ == '__main__':
    print('start from here')
    from evaluate import plot_tr_val_loss, model_basin_performance, model_grid_performance

    result_path = record_training_info()
    model_basin_performance(model=model, dataset=tr_dataset, save_path=result_path + 'basin_perf.png')
    model_grid_performance(model=model, dataset=test_dataset, save_path=result_path + 'grid_perf.png')
