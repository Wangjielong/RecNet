import numpy as np
from grace_gfo.jpl_mascon import JPL_Mascon
from models import gldas
from temp import ghcn_cams
from metrics import corr
import matplotlib.pyplot as plt

jpl = JPL_Mascon()
shp_path = 'data/Yangtze/Yangtze vector/长江流域范围矢量图.shp'

graceSeries_dict, gfoSeries_dict = jpl.getRegionSeries(interp='linear', shp_path=shp_path)

grace_time = np.array([key for key in graceSeries_dict.keys()])[:, np.newaxis]  # 2002-04~2017-06
grace_series = np.array([value for value in graceSeries_dict.values()])[:, np.newaxis]

# 2018-06~2021-12
gfo_time = np.array([key for key in gfoSeries_dict.keys() if key <= np.datetime64('2021-12')])[:, np.newaxis]
gfo_series = np.array([value for key, value in gfoSeries_dict.items()
                       if key <= np.datetime64('2021-12')])[:, np.newaxis]


def lagged_corr(var_name, lags=12):
    if var_name == 'P':  # precip
        gldas_o = gldas.MultiGldas('data/GLDAS/')
        precip_dict = gldas_o.getRegionAnomalySeries(comp_str='Precip', start='2002-04', end='2021-12')
        series = np.array([value for value in precip_dict.values()])[:, np.newaxis]
    elif var_name == 'TWS':
        gldas_o = gldas.MultiGldas('data/GLDAS/')
        tws_dict = gldas_o.getRegionAnomalySeries(comp_str='TWS', start='2002-04', end='2021-12')
        series = np.array([value for value in tws_dict.values()])[:, np.newaxis]
    elif var_name == 'T':  # temperature
        temp_o = ghcn_cams.GhcnCams()
        temp_dict = temp_o.getRegionSeries(shp_path=shp_path)
        temp_mean = np.mean(
            np.array([value for key, value in temp_dict.items() if
                      np.datetime64('2004-01') <= key <= np.datetime64('2009-12')]))
        series = np.array([value for key, value in temp_dict.items() if
                           np.datetime64('2002-04') <= key <= np.datetime64('2021-12')])[:, np.newaxis] - temp_mean
    else:
        raise ValueError('invalid variable name')

    grace_len, gfo_len = len(grace_series), len(gfo_series)
    series_grace = series[:grace_len]
    series_gfo = series[-gfo_len:]

    corr_list = [corr(np.vstack((series_grace, series_gfo)), np.vstack((grace_series, gfo_series)))]

    for lag in range(1, lags):
        lagged_series_grace = series_grace[:-lag]
        lagged_series_gfo = series_gfo[:-lag]
        lagged_grace_series = grace_series[lag:]
        lagged_gfo_series = gfo_series[lag:]
        corr_list.append(corr(np.vstack((lagged_series_grace, lagged_series_gfo)),
                              np.vstack((lagged_grace_series, lagged_gfo_series))))

    fig, ax = plt.subplots(1, 1)
    label = ['lag' + str(value) for value in range(lags)]
    ax.bar(label, corr_list)
    plt.title(var_name)
    plt.show()

    return corr_list


if __name__ == '__main__':
    print('start from here')
    t = lagged_corr(var_name='T')
