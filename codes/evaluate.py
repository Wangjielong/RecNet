import matplotlib.pyplot as plt
import numpy as np
from metrics import corr, nse, mae, rmse


def plot_tr_val_loss(tr_loss, val_loss, save_path=None, plot_show=True):
    best_val_loss = min(val_loss)
    num_neglect = int(len(tr_loss) * 0.05)
    fig, ax = plt.subplots(1, 1)
    ax.plot(range(len(tr_loss) - num_neglect), tr_loss[num_neglect:], label='training', color='grey', lw=2)
    ax.plot(range(len(val_loss) - num_neglect), val_loss[num_neglect:], label='validation', color='b', lw=2)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('MSE [cm]')
    plt.title(f'Best val loss = {best_val_loss}')
    plt.grid()

    if save_path:
        plt.savefig(save_path)
    if plot_show:
        plt.show()
    return fig


def model_basin_performance(model, dataset, plot_flag=True, save_path=None, plot_show=True):
    model.to('cpu')
    model.eval()
    input_grid_t = dataset.input_grid
    target_grid_t = dataset.target_grid
    output_grid_t = model(input_grid_t)

    target_grid_a = target_grid_t.detach().cpu().numpy().squeeze()
    output_grid_a = output_grid_t.detach().cpu().numpy().squeeze()

    data_weight = np.load('data/gridWeight_05.npz', allow_pickle=True)
    grid_weight_05 = data_weight['gridWeight_05']  # 32*80
    total_weight = np.sum(grid_weight_05)

    model_basin_average = np.array([np.sum(grid * grid_weight_05) / total_weight for grid in output_grid_a])
    target_basin_average = np.array([np.sum(grid * grid_weight_05) / total_weight for grid in target_grid_a])

    CC = corr(model_basin_average, target_basin_average)
    RMSE = rmse(model_basin_average, target_basin_average)
    NSE = nse(model_basin_average, target_basin_average)

    if plot_flag:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.plot(range(len(model_basin_average)), model_basin_average, c='r', label='model', lw=2)
        ax.plot(range(len(target_basin_average)), target_basin_average, c='b', label='grace/gfo', lw=2)
        plt.title(f'CC={CC:.2f} RMSE={RMSE:.2f} NSE={NSE:.2f} MSE={RMSE ** 2:.2f}')
        plt.legend()
        plt.grid()
        if save_path:
            plt.savefig(save_path)
    if plot_show:
        plt.show()
    return CC, NSE, RMSE


def model_grid_performance(model, dataset, plot_flag=True, save_path=None, plot_show=True):
    model.to('cpu')
    model.eval()
    input_grid_t = dataset.input_grid
    target_grid_t = dataset.target_grid
    output_grid_t = model(input_grid_t)

    mask_data = np.load('data/mask32x80_a.npz', allow_pickle=True)
    mask_a = mask_data['mask32x80_a'].astype(float)
    mask_a[mask_a < 1] = np.nan

    target_grid_a = target_grid_t.detach().cpu().numpy().squeeze()
    output_grid_a = output_grid_t.detach().cpu().numpy().squeeze()

    CC = corr(target_grid_a, output_grid_a) * mask_a
    MAE = mae(target_grid_a, output_grid_a) * mask_a
    NSE = nse(target=target_grid_a, prediction=output_grid_a) * mask_a

    if plot_flag:
        fig, ax = plt.subplots(3, 1, figsize=(8, 8))

        i0 = ax[0].imshow(CC, cmap='jet')
        ax[0].set_title('CC')
        fig.colorbar(i0, ax=ax[0])
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        i1 = ax[1].imshow(NSE, cmap='jet', vmin=-1)
        fig.colorbar(i1, ax=ax[1])
        ax[1].set_title('NSE')
        ax[1].set_xticks([])
        ax[1].set_yticks([])

        i2 = ax[2].imshow(MAE, cmap='jet', vmax=8)
        fig.colorbar(i2, ax=ax[2])
        ax[2].set_title('MAE')
        ax[2].set_xticks([])
        ax[2].set_yticks([])

        if save_path:
            plt.savefig(save_path)

    if plot_show:
        plt.show()
    return CC, NSE, MAE


if __name__ == '__main__':
    print('start from here')
