from unet import UNet
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from util import shp

torch.set_default_dtype(torch.float64)


def get_basin_grid_weight(resolution, shp_file):
    """using the .shp file of sub-basins to calculate and crop the grid weight """
    region_mask, *others = shp.regionMask(res=resolution, shp_file=shp_file)
    slice_list = [slice(103, 135), slice(163, 243)]  # shape: (32,80)
    region_mask = region_mask[tuple(slice_list)]  # crop

    data_weight = np.load('data/gridWeight_05.npz', allow_pickle=True)
    grid_weight = data_weight['gridWeight_05']  # 32*80
    grid_weight *= region_mask
    total_weight = np.sum(grid_weight)
    return grid_weight, total_weight


def reconstruct_basin_twsa(model_path, shp_file, plot_flag=True, save_fig=None, plot_show=True):
    """using the trained model to reconstruct the TWSA over the Yangtze river basin"""
    model_init_dict = {'in_channels': 3, 'n_classes': 1, 'depth': 5, 'wf': 4}
    model = UNet(**model_init_dict)
    model.to('cpu')
    model.eval()
    model.load_state_dict(torch.load(model_path))

    # 1971/01~2002/03    2017/07~2018/05
    # rec for reconstruction
    rec_input = np.load('data/rec_input.npz')
    rec_t, rec_s, rec_g = rec_input['time'], rec_input['series'], rec_input['grid']

    # 2002/04~2017/06    2018/06~2021/12
    # tr for training
    tr_input = np.load('data/tr_input.npz')
    tr_t, tr_s, tr_g = tr_input['time'], tr_input['series'], tr_input['grid']

    target = np.load('data/target.npz')
    target_t, target_s, target_g = target['time'], target['series'], target['grid']
    target_grid_a = target_g.squeeze()

    idx1, idx2 = 375, 183
    all_t = np.concatenate((rec_t[:idx1], tr_t[:idx2], rec_t[idx1:], tr_t[idx2:]), axis=0)  # 612*1
    all_g = np.concatenate((rec_g[:idx1], tr_g[:idx2], rec_g[idx1:], tr_g[idx2:]), axis=0)  # 612*3*32*80
    # normalization
    grid_mean_std_data = np.load('data/grid_mean_std.npz')
    grid_mean, grid_std = grid_mean_std_data['mean'], grid_mean_std_data['std']
    all_g = (all_g - grid_mean.reshape(1, -1, 1, 1)) / grid_std.reshape(1, -1, 1, 1)

    grid_weight, total_weight = get_basin_grid_weight(resolution=0.5, shp_file=shp_file)

    output_grid_t = model(torch.from_numpy(all_g))  # 612*1*32*80
    output_grid_a = output_grid_t.detach().cpu().numpy().squeeze()

    model_basin_average = np.array([np.sum(grid * grid_weight) / total_weight for grid in output_grid_a])
    target_basin_average = np.array([np.sum(grid * grid_weight) / total_weight for grid in target_grid_a])

    if plot_flag:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.plot(all_t, model_basin_average, c='r', label='model', lw=2)
        ax.plot(target_t, target_basin_average, c='b', label='GRACE/GFO', lw=2)
        plt.legend()
        plt.grid()
        if save_fig:
            plt.savefig(save_fig)
        if plot_show:
            plt.show()

    return all_t, model_basin_average, target_t, target_basin_average, output_grid_a, target_grid_a


def get_basin_ensemble_twsa(models_path, shp_file, plot_flag=True, save_fig=False, plot_show=True):
    """
    average the results of k-fold cross validation
    models_path: the path that contains all folds
    """
    folds_list = os.listdir(models_path)
    ensemble_list = []
    ensemble_grid_list = []
    all_t, target_t, target_twsa, target_grid_a = None, None, None, None
    for fold in folds_list:
        model_path = models_path + fold + '/model.pt'
        all_t, model_twsa, target_t, target_twsa, model_grid_a, target_grid_a = reconstruct_basin_twsa(
            model_path=model_path,
            shp_file=shp_file,
            plot_show=False,
            plot_flag=False)
        ensemble_list.append(model_twsa)
        ensemble_grid_list.append(model_grid_a)
        print(f'fold={fold}')

    model_grid_a = np.mean(np.array(ensemble_grid_list), axis=0)
    ensemble_a = np.array(ensemble_list)  # num_folds*612
    twsa_std = np.std(ensemble_a, axis=0)
    twsa_mean = np.mean(ensemble_a, axis=0)
    all_t = all_t.squeeze()

    if plot_flag:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.fill_between(all_t, twsa_mean - 1.96 * twsa_std, twsa_mean + 1.96 * twsa_std)
        ax.plot(all_t, twsa_mean, c='r', label='model', lw=2)
        ax.plot(target_t, target_twsa, c='b', label='GRACE/GFO', lw=2)
        plt.legend()
        plt.grid()

        if save_fig:
            plt.savefig(save_fig)
    if plot_show:
        plt.show()

    return all_t, twsa_mean, twsa_std, target_t, target_twsa, model_grid_a, target_grid_a


if __name__ == '__main__':
    print('start from here')
    shp_path = r'D:\Doctoral Documents\program\data\Yangtze\yangtze vector\长江流域范围矢量图.shp'
    shp_path2 = r'D:\Doctoral Documents\program\data\Yangtze\vector data\2流域\长江流域\JSJ.shp'
    # all_t, twsa_mean, twsa_std, target_t, target_twsa, model_grid_mean, target_grid_a = get_basin_ensemble_twsa(
    #     models_path='cross_validation/', shp_file=shp_path)
    # np.savez('data/rec_grace_gfo.npz', all_t=all_t, twsa_mean=twsa_mean, twsa_std=twsa_std, target_t=target_t,
    #          target_twsa=target_twsa, model_grid=model_grid_mean, target_grid=target_grid_a)
