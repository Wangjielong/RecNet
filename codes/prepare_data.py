# This python script aims to prepare the data for our model inputs and outputs, including cropping the data
# and saving the data to memory in order to speed up data fetch.
# The time coverage of GRACE is from 2002-04 to 2017-06 while GFO is from 2018-06 to 2021-12, with a spatial
# resolution of 0.5deg
# we used the precipitation and temperature of previous month and GLDAS TWSA of current month to
# reconstruct the Yangtze TWSA during the past 50 years (i.e., from 1971-01~2021-12)
# GRACE is from 2002-04  ~ 2017-06   GFO is  from 2018-06 ~ 2021-12    data gap is 2017-07 ~ 2018-05
# Note that:

# This python script isn't run in the venv as we use many packages to prepare data
# and is run under the program directory
import torch

from temp import ghcn_cams
from models import gldas
from grace_gfo.jpl_mascon import JPL_Mascon
from climate import climate_indices

import numpy as np
import torch.nn as nn

shp_path = 'data/Yangtze/Yangtze vector/长江流域范围矢量图.shp'
# the time period that we to reconstruct
start_time, end_time = np.datetime64('1971-01'), np.datetime64('2021-12')
# slice list for 0.5deg data
slice_list = [slice(103, 135), slice(163, 243)]  # shape: (32,80)
mask_a = np.load(r'D:\Doctoral Documents\program\paper\reconstructionV2\data\mask32x80_a.npz',
                 allow_pickle=True)['mask32x80_a']


def prepare_temperature(lag=1):
    temp_o = ghcn_cams.GhcnCams(filepath='data/GHCN_CAMS_Temperature/air.mon.mean.nc')
    temp_dict = temp_o.getRegionGrid(shp_path=shp_path)
    # Note that we used the temperature of previous month as input
    # select and crop data
    cropped_grid_dict = {key: value[tuple(slice_list)] * mask_a for key, value in
                         temp_dict.items() if start_time - lag <= key <= end_time - lag}

    baseline_grid_dict = {key: value for key, value in cropped_grid_dict.items()
                          if np.datetime64('2004-01') <= key <= np.datetime64('2009-12')}
    baseline_grid = np.mean(np.array([grid for grid in baseline_grid_dict.values()]), axis=0)

    data_time = np.array([key for key in cropped_grid_dict.keys()])[:, np.newaxis]  # 1970-12~2021-11 when lag=1
    data_a = np.array([value for value in cropped_grid_dict.values()])[:, np.newaxis, :, :]  # 612*1*32*80
    # deduct the average field
    data_a -= baseline_grid
    data_a[np.isnan(data_a)] = 0.0
    return data_time, data_a


def prepare_precipitation_tws(var_name='Precip'):
    """
    we use the precipitation of GLDAS2.0 from 1970-12 to 1999-12 and
    precipitation of GLDAS2.1 from 2000-01 to 2021-11
    But GLDAS TWS has no lag
    :param var_name: Precip or TWS
    :return:
    """
    gldas_020 = gldas.MultiGldas(fileDirectory=r'data/GLDAS_V2.0/')
    gldas_021 = gldas.MultiGldas(fileDirectory=r'data/GLDAS/')
    # Note that we used the precipitation of previous month as input
    # the average field is already deducted

    if var_name == 'Precip':
        data_020_dict = gldas_020.getAnomalyGrid(comp_str=var_name, start='1970-12', end='1999-12')
        data_021_dict = gldas_021.getAnomalyGrid(comp_str=var_name, start='2000-01', end='2021-11')
    elif var_name == 'TWS':
        data_020_dict = gldas_020.getAnomalyGrid(comp_str=var_name, start='1971-01', end='1999-12')
        data_021_dict = gldas_021.getAnomalyGrid(comp_str=var_name, start='2000-01', end='2021-12')
    else:
        raise ValueError('invalid variable name')

    data_020_a = np.array([grid for grid in data_020_dict.values()])
    data_021_a = np.array([grid for grid in data_021_dict.values()])

    data_time_020 = np.array([key for key in data_020_dict.keys()])[:, np.newaxis]
    data_time_021 = np.array([key for key in data_021_dict.keys()])[:, np.newaxis]

    data_time = np.vstack((data_time_020, data_time_021))
    # 0.25deg
    data_a = np.concatenate((data_020_a, data_021_a), axis=0)[:, np.newaxis, :, :]  # 612*1*720*1440

    data_t = torch.from_numpy(data_a)
    avgpool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
    data_t = avgpool(data_t)  # downscale to 0.5deg
    data_a = data_t.numpy()  # 612*1*360*720
    data_a = data_a[:, :, 103:135, 163:243] * mask_a  # 612*1*32*80
    data_a[np.isnan(data_a)] = 0.0
    return data_time, data_a


def prepare_grace_gfo():
    jpl = JPL_Mascon()
    graceGrid_dict, gfoGrid_dict = jpl.getRegionGrid(interp='linear', shp_path=shp_path)
    graceSeries_dict, gfoSeries_dict = jpl.getRegionSeries(interp='linear', shp_path=shp_path)

    grace_time = np.array([key for key in graceSeries_dict.keys()])[:, np.newaxis]  # 2002-04~2017-06
    grace_series = np.array([value for value in graceSeries_dict.values()])[:, np.newaxis]
    grace_grid = np.array([value[tuple(slice_list)] for value in graceGrid_dict.values()])  # 183*32*80
    grace_grid[np.isnan(grace_grid)] = 0.0

    # 2018-06~2021-12
    gfo_time = np.array([key for key in gfoSeries_dict.keys() if key <= end_time])[:, np.newaxis]
    gfo_series = np.array([value for key, value in gfoSeries_dict.items() if key <= end_time])[:, np.newaxis]
    # 43*32*80
    gfo_grid = np.array([value[tuple(slice_list)] for key, value in gfoGrid_dict.items() if key <= end_time])
    gfo_grid[np.isnan(gfo_grid)] = 0.0

    return grace_time, grace_series, grace_grid, gfo_time, gfo_series, gfo_grid


def prepare_climate_indices(index_name='Nino34'):
    """
    preparing the climate indices including nino34,pdo,dmi and nao
    we set a conservative and constant lag of 12
    :param index_name: the name of the climate index
    :return:
    """
    lags = 12
    # since the lag is fixed, so it starts from 1970-01
    data_time = np.arange(np.datetime64('1970-01'), end_time + 1, np.timedelta64(1, str('M')))[:, np.newaxis]

    if index_name == 'Nino34':
        data_pd = climate_indices.read_nino34(start_time='1970-01', end_time='2021-12')
    elif index_name == 'NAO':
        data_pd = climate_indices.read_nao(start_time='1970-01', end_time='2021-12')
    elif index_name == 'DMI':
        data_pd = climate_indices.read_dmi(start_time='1970-01', end_time='2021-12')
    elif index_name == 'PDO':
        data_pd = climate_indices.read_pdo(start_time='1970-01', end_time='2021-12')
    else:
        raise ValueError('must be Nino34, NAO, DMI,or PDO')
    data_a = np.array(data_pd.values)  # 624*1

    time_list = []
    data_list = []
    for lag in range(lags):
        time_list.append(data_time[lag:-lags + lag])
        data_list.append(data_a[lag:-lags + lag])
    time_list.append(data_time[lags:])
    data_list.append(data_a[lags:])

    input_time = np.array(time_list).swapaxes(0, 2).squeeze()  # 612*13
    input_data = np.array(data_list).swapaxes(0, 2).squeeze()  # 612*13

    return input_time, input_data


def prepare_target(save_flag=False):
    grace_time, grace_series, grace_grid, gfo_time, gfo_series, gfo_grid = prepare_grace_gfo()
    time = np.concatenate((grace_time, gfo_time), axis=0)  # 226*1
    series = np.concatenate((grace_series, gfo_series), axis=0)  # 226*1
    grid = np.concatenate((grace_grid, gfo_grid), axis=0)[:, np.newaxis, :, :]  # 226*1*32*80

    if save_flag:
        np.savez_compressed('target', time=time, series=series, grid=grid)

    return time, series, grid


def prepare_input(save_flag=False):
    _, temp_grid = prepare_temperature(lag=1)
    _, precip_grid = prepare_precipitation_tws(var_name='Precip')  # lag=1
    _, gldas_twsa_grid = prepare_precipitation_tws(var_name='TWS')  # lag=0

    input_grid = np.concatenate((precip_grid, temp_grid, gldas_twsa_grid), axis=1)  # 612*3*32*80
    _, input_series = prepare_climate_indices(index_name='Nino34')  # shape=(612,13)

    # using the time to split the input
    # 1971/01~2002/04~2017/06~2018/06~2021/12
    #          idx1     idx2    idx3
    input_time = np.arange(start_time, end_time + 1, np.timedelta64(1, str('M')))  # 612*1
    t1 = np.datetime64('2002-04')
    idx1 = int(np.where(input_time == t1)[0])  # 375

    t2 = np.datetime64('2017-06')
    idx2 = int(np.where(input_time == t2)[0])  # 557

    t3 = np.datetime64('2018-06')
    idx3 = int(np.where(input_time == t3)[0])  # 569

    # 1971/01~2002/03    2017/07~2018/05
    # rec for reconstruction
    rec_input_time = np.concatenate((input_time[:idx1], input_time[idx2 + 1:idx3]), axis=0)[:, np.newaxis]  # 386*1
    rec_input_grid = np.concatenate((input_grid[:idx1], input_grid[idx2 + 1:idx3]), axis=0)  # 386*3*32*80
    rec_input_series = np.concatenate((input_series[:idx1], input_series[idx2 + 1:idx3]), axis=0)  # 386*13

    # 2002/04~2017/06    2018/06~2021/12
    # tr for training
    tr_input_time = np.concatenate((input_time[idx1:idx2 + 1], input_time[idx3:]), axis=0)[:, np.newaxis]  # 226*1
    tr_input_grid = np.concatenate((input_grid[idx1:idx2 + 1], input_grid[idx3:]), axis=0)  # 226*3*32*80
    tr_input_series = np.concatenate((input_series[idx1:idx2 + 1], input_series[idx3:]), axis=0)  # 226*13

    if save_flag:
        np.savez_compressed('rec_input', time=rec_input_time, series=rec_input_series, grid=rec_input_grid)
        np.savez_compressed('tr_input', time=tr_input_time, series=tr_input_series, grid=tr_input_grid)

    return tr_input_time, tr_input_series, tr_input_grid, rec_input_time, rec_input_series, rec_input_grid


if __name__ == '__main__':
    print('start')
    t, d = prepare_precipitation_tws(var_name='Precip')
    data = d.squeeze()

    data_weight = np.load(r'D:\Doctoral Documents\program\paper\reconstructionV2\data\gridWeight_05.npz',
                          allow_pickle=True)
    grid_weight_05 = data_weight['gridWeight_05']  # 32*80
    total_weight = np.sum(grid_weight_05)

    basin_average = np.array([np.sum(grid * grid_weight_05) / total_weight for grid in data])
