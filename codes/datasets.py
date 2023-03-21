import numpy as np
import functools
import torchvision.transforms.functional as TF
import torch
from torch.utils.data import Dataset


def preprocMetaData(input_time):
    """
    preprocessing the metadata including latitude, longitude, cosine and sine of month index
    Those scalar variables are converted to flat gridded channels at the data's spatial size, which is 128 x 320.
    Then these gridded variables are stacked along the channel dimension
    :return:gridded data
    """

    cosineOfMonth = np.array([np.cos(2 * np.pi * int(str(year_month[0])[-2:]) / 12) for year_month in input_time])
    sineOfMonth = np.array([np.sin(2 * np.pi * int(str(year_month[0])[-2:]) / 12) for year_month in input_time])

    cosineOfMonth = cosineOfMonth.reshape(len(cosineOfMonth), 1, 1) * np.ones(
        (len(cosineOfMonth), 32, 80))  # 612*128*320
    sineOfMonth = sineOfMonth.reshape(len(sineOfMonth), 1, 1) * np.ones(
        (len(sineOfMonth), 32, 80))  # 612*128*320

    # adding the channel dimension
    cosineOfMonth, sineOfMonth = cosineOfMonth[:, np.newaxis, :, :], sineOfMonth[:, np.newaxis, :, :]
    metadata = np.concatenate((cosineOfMonth, sineOfMonth), axis=1)  # 612*4*128*320

    return input_time, metadata


def get_input_mean_std(save_flag=False):
    """we estimate the mean and std values on all the data we have"""
    rec_data = np.load('data/rec_input.npz', allow_pickle=True)
    tr_data = np.load('data/tr_input.npz', allow_pickle=True)

    rec_series, rec_grid = rec_data['series'], rec_data['grid']
    tr_series, tr_grid = tr_data['series'], tr_data['grid']

    series = np.concatenate((rec_series, tr_series), axis=0)
    grid = np.concatenate((rec_grid, tr_grid), axis=0)

    grid_mean, grid_std = np.mean(grid, axis=(0, 2, 3)), np.std(grid, axis=(0, 2, 3))
    series_mean, series_std = np.mean(series, axis=0), np.std(series, axis=0)

    if save_flag:
        np.savez_compressed('data/grid_mean_std', mean=grid_mean, std=grid_std)
        np.savez_compressed('data/series_mean_std', mean=series_mean, std=series_std)

    return series_mean, series_std, grid_mean, grid_std


@functools.lru_cache(1)
def get_training_target(filepath='data/target.npz'):
    target_data = np.load(filepath)
    target_time = target_data['time']  # 226*1
    target_series = target_data['series']  # 226*1
    target_grid = target_data['grid']  # 226*1*32*80
    return target_time, target_series, target_grid


@functools.lru_cache(1)
def get_training_input(filepath='data/tr_input.npz', normalization_bool=True):
    """
    If normalization_bool==True, we normalize the input on all the data we have
    """
    input_data = np.load(filepath)
    input_time = input_data['time']  # 226*1
    input_series = input_data['series']  # 226*13
    input_grid = input_data['grid']  # 226*3*32*80

    if normalization_bool:
        grid_mean_std_data = np.load('data/grid_mean_std.npz')
        series_mean_std_data = np.load('data/series_mean_std.npz')

        grid_mean, grid_std = grid_mean_std_data['mean'], grid_mean_std_data['std']
        series_mean, series_std = series_mean_std_data['mean'], series_mean_std_data['std']
        input_grid = (input_grid - grid_mean.reshape(1, -1, 1, 1)) / grid_std.reshape(1, -1, 1, 1)
        input_series = (input_series - series_mean.reshape(1, -1)) / series_std.reshape(1, -1)

    return input_time, input_series, input_grid


def augment_training_data(input_grid, target_grid):
    """
    augmenting training data
    :param input_grid: 4D tensor, N*C*H*W
    :param target_grid: 4D tensor, N*C*H*W
    :return:
    """
    # record the original data
    temp_input, temp_target = input_grid, target_grid
    # Horizontally flip the given data
    # input_grid = torch.cat([input_grid, TF.hflip(temp_input)], dim=0)
    # target_grid = torch.cat([target_grid, TF.hflip(temp_target)], dim=0)
    #
    # # Vertically flip the given data
    # input_grid = torch.cat([input_grid, TF.vflip(temp_input)], dim=0)
    # target_grid = torch.cat([target_grid, TF.vflip(temp_target)], dim=0)
    #
    # # Horizontally and vertically flip the given data
    # input_grid = torch.cat([input_grid, TF.vflip(TF.hflip(temp_input))], dim=0)
    # target_grid = torch.cat([target_grid, TF.vflip(TF.hflip(temp_target))], dim=0)

    # adding the noise
    # noise = np.random.normal(loc=0, scale=0.3, size=(32, 80))
    # input_grid = torch.cat([input_grid, temp_input + noise])
    # target_grid = torch.cat([target_grid, temp_target])

    # rotate
    input_grid = torch.cat([input_grid, TF.rotate(temp_input, angle=3.0)], dim=0)
    target_grid = torch.cat([target_grid, TF.rotate(temp_target, angle=3.0)], dim=0)

    return input_grid, target_grid


input_t, input_s, input_g = get_training_input(normalization_bool=True)
target_t, target_s, target_g = get_training_target()
data_length = len(target_t)


class RecDataset(Dataset):
    def __init__(self, idx_list, is_train=True, augmentation_bool=False):
        """select the data using the idx_list"""
        self.input_time, self.target_time = input_t[idx_list], target_t[idx_list]
        self.input_series = torch.from_numpy(input_s[idx_list])
        self.input_grid = torch.from_numpy(input_g[idx_list])

        self.target_series = torch.from_numpy(target_s[idx_list])
        self.target_grid = torch.from_numpy(target_g[idx_list])

        self.month_index = torch.from_numpy(self.get_month_index())

        if is_train and augmentation_bool:
            print('augmenting...')
            self.input_grid, self.target_grid = augment_training_data(self.input_grid, self.target_grid)
            temp_input_series, temp_target_series = self.input_series, self.target_series,
            temp_month_index = self.month_index
            while len(self.month_index) < len(self.input_grid):
                self.month_index = torch.cat([self.month_index, temp_month_index], dim=0)
                self.input_series = torch.cat([self.input_series, temp_input_series], dim=0)
                self.target_series = torch.cat([self.target_series, temp_target_series], dim=0)

    def __len__(self):
        return len(self.input_grid)

    def get_month_index(self):
        return np.array([float(str(year_month[0])[-2:]) for year_month in self.target_time]).reshape(-1, 1)

    def __getitem__(self, idx):
        return (self.month_index[idx], self.input_series[idx], self.input_grid[idx],
                self.target_series[idx], self.target_grid[idx])


def generate_tr_val_test_idx(data_len=226, val_ratio=0.1, test_ratio=0.1, shuffle=True):
    indices = range(data_len)
    if shuffle:
        # set a seed for reproducibility. Since you are generating random values, setting a seed
        # will ensure that the values generated are the same if the seed set is the same each time the code is run
        np.random.seed(2)
        indices = np.random.permutation(range(data_len))

    split_idx1 = int(len(indices) * (1 - val_ratio - test_ratio))
    split_idx2 = int(len(indices) * (1 - test_ratio))

    tr_idx = list(indices[:split_idx1])
    val_idx = list(indices[split_idx1:split_idx2])
    test_idx = list(indices[split_idx2:])
    return tr_idx, val_idx, test_idx


if __name__ == '__main__':
    print('start')

    print('over')
