import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from util import stl
import pandas as pd
from scipy.signal import savgol_filter as sf

plt.rc('font', family='sans-serif')


def plot_cross_validation_loss(data_path='cross_validation/', show_flag=True, save_flag=False):
    params = {
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'font.size': 12,
        'legend.fontsize': 10,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'axes.linewidth': 1,
        'figure.facecolor': 'w',
    }

    mpl.rcParams.update(params)

    fig, axes = plt.subplots(2, 5, figsize=(12, 6), layout='constrained')
    title_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    for idx, ax in enumerate(axes.flat):
        loss = np.load(data_path + str(idx) + '/loss.npz', allow_pickle=True)
        tr_loss, val_loss = loss['tr_loss'], loss['val_loss']
        ax.plot(np.arange(len(tr_loss)), tr_loss, label='training loss', c='blue', lw=2)
        ax.plot(np.arange(len(val_loss)), val_loss, label='validation loss', c='r', lw=2)
        ax.grid()

        ax.tick_params(labelsize=10)

        ax.text(s=title_list[idx], fontweight='bold',
                x=0.12, y=0.94, transform=ax.transAxes, horizontalalignment='right', fontsize=12)

        if idx == 0 or idx == 5:
            ax.set_ylabel('MSE [cm^2]')
        if idx >= 5:
            ax.set_xlabel('Epoch')

    plt.legend(loc='upper right')

    if save_flag:
        plt.savefig('figures/FigS2.jpg', dpi=800)

    if show_flag:
        plt.show()


def plot_rec_grace_gfo(show_flag=True, save_flag=False):
    params = {
        'axes.labelsize': 15,
        'axes.titlesize': 12,
        'font.size': 15,
        'legend.fontsize': 15,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'axes.linewidth': 1.5,
        'figure.facecolor': 'w',
    }

    mpl.rcParams.update(params)
    # -----------------------------------------------------------prepare data
    mask_data = np.load('data/mask32x80_a.npz', allow_pickle=True)
    mask_a = mask_data['mask32x80_a']
    mask_a = np.where(mask_a, mask_a, np.nan)

    data = np.load('data/rec_grace_gfo.npz', allow_pickle=True)
    all_t, twsa_mean, twsa_std, target_t, target_twsa = data['all_t'], data['twsa_mean'], data['twsa_std'], data[
        'target_t'], data['target_twsa']

    model_grid = data['model_grid'] * mask_a  # the grid of model output 612*32*80
    target_grid = data['target_grid'] * mask_a  # 226*32*80

    model_grid = model_grid[:, :, 15:]  # crop the grid
    target_grid = target_grid[:, :, 15:]
    mask_a = mask_a[:, 15:]
    mask_a[np.isnan(mask_a)] = 0.0

    # select 2004-08, 2012-08, and 2020-08
    t11, t12, t13 = np.datetime64('2004-01'), np.datetime64('2012-08'), np.datetime64('2020-04')
    idx_model_1, idx_model_2, idx_model_3 = int(np.where(all_t == t11)[0]), int(np.where(all_t == t12)[0]), int(
        np.where(all_t == t13)[0])
    idx_target_1, idx_target_2, idx_target_3 = int(np.where(target_t == t11)[0]), int(
        np.where(target_t == t12)[0]), int(np.where(target_t == t13)[0])

    t1 = np.datetime64('2002-04')
    idx1 = int(np.where(all_t == t1)[0])  # 375

    t2 = np.datetime64('2017-06')
    idx2 = int(np.where(target_t == t2)[0])  # 182

    rec_t, rec_a = all_t[idx1:], twsa_mean[idx1:]
    rec_std = twsa_std[idx1:]
    grace_t, grace_a = target_t[:idx2 + 1], target_twsa[:idx2 + 1]
    gfo_t, gfo_a = target_t[idx2 + 1:], target_twsa[idx2 + 1:]

    # ------------------------------------------------------------------------- plotting
    fig = plt.figure(layout="constrained", figsize=(14, 12))
    gs = fig.add_gridspec(nrows=5, ncols=3)
    ax0 = fig.add_subplot(gs[0:2, :])
    # series plotting
    ax0.plot(gfo_t, gfo_a, 'bo', label='GRACE-FO', lw=3, markersize=7, ls='-')
    ax0.plot(grace_t, grace_a, c='darkgreen', label='GRACE', lw=3, ls='-', markersize=7, marker='s')
    ax0.plot(rec_t, rec_a, c='red', label='RecNet', lw=3)
    ax0.fill_between(rec_t, rec_a - 3 * rec_std, rec_a + 3 * rec_std, label='Bound', color='gray', alpha=0.5)
    ax0.text(s='a', fontweight='bold',
             x=0.98, y=0.04, transform=ax0.transAxes, horizontalalignment='right', fontsize=15)
    ax0.legend()
    ax0.grid()
    ax0.set_ylabel('TWSA [cm]')

    # spatial plotting

    cmap = 'seismic'
    levels = 100
    getattr(mpl.cm, cmap).set_bad(color=np.array([0.95, 0.95, 0.95]))
    locator = mpl.ticker.MultipleLocator(15)
    formatter = mpl.ticker.StrMethodFormatter('{x:.0f}')

    model_idx = [idx_model_1, idx_model_2, idx_model_3]
    target_idx = [idx_target_1, idx_target_2, idx_target_3]
    ax1 = list([fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1]), fig.add_subplot(gs[2, 2])])
    ax2 = list([fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1]), fig.add_subplot(gs[3, 2])])
    ax3 = list([fig.add_subplot(gs[4, 0]), fig.add_subplot(gs[4, 1]), fig.add_subplot(gs[4, 2])])

    target_label = ['b', 'c', 'd']
    model_label = ['e', 'f', 'g']
    diff_label = ['h', 'i', 'j']
    title_label = ['Jan 2004', 'Aug 2012', 'Apr 2020']
    im_target, im_model = None, None
    # Target
    for col in range(3):
        ax1[col].text(s=target_label[col], fontweight='bold', x=0.98, y=0.04,
                      transform=ax1[col].transAxes, horizontalalignment='right', fontsize=15)

        ax1[col].text(s=title_label[col], x=0.6, y=0.9,
                      transform=ax1[col].transAxes, horizontalalignment='right', fontsize=15)

        im_target = ax1[col].contourf(np.flipud(target_grid[target_idx[col]]), cmap=cmap, levels=levels)
        ax1[col].set_xticks([])
        if col == 0:
            ax1[col].set_yticklabels(np.arange(24, 40, 4))
        else:
            ax1[col].set_yticks([])
        ax1[col].contour(np.flipud(mask_a), linewidths=0.8, levels=1)

    cbar = fig.colorbar(im_target, ax=ax1, orientation='vertical', ticks=locator, format=formatter, pad=0.02,
                        extend='both')
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Observed [cm]', fontsize=14, fontweight='bold')
    # Model
    for col in range(3):
        ax2[col].text(s=model_label[col], fontweight='bold', x=0.98, y=0.04,
                      transform=ax2[col].transAxes, horizontalalignment='right', fontsize=15)
        im_model = ax2[col].contourf(np.flipud(model_grid[model_idx[col]]), cmap=cmap, levels=levels)

        ax2[col].set_xticks([])
        if col == 0:
            ax2[col].set_yticklabels(np.arange(24, 40, 4))
        else:
            ax2[col].set_yticks([])
        ax2[col].contour(np.flipud(mask_a), linewidths=0.8, levels=1)

    cbar = fig.colorbar(im_model, ax=ax2, orientation='vertical', ticks=locator, format=formatter, pad=0.02)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Reconstructed [cm]', fontsize=14, fontweight='bold')
    # difference
    for col in range(3):
        ax3[col].text(s=diff_label[col], fontweight='bold', x=0.98, y=0.04,
                      transform=ax3[col].transAxes, horizontalalignment='right', fontsize=15)
        im_model = ax3[col].contourf(np.flipud(np.abs(model_grid[model_idx[col]] - target_grid[target_idx[col]])),
                                     cmap='rainbow_r', levels=6)
        ax3[col].set_xticklabels(np.arange(90, 122, 10))
        if col > 0:
            ax3[col].set_yticks([])
        else:
            ax3[col].set_yticklabels(np.arange(24, 40, 4))
        ax3[col].contour(np.flipud(mask_a), linewidths=0.8, levels=1)

    cbar = fig.colorbar(im_model, ax=ax3, orientation='vertical', format=formatter, pad=0.02)
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label('Difference [cm]', fontsize=14, fontweight='bold', labelpad=15)

    if save_flag:
        plt.savefig('figures/Fig3.jpg', dpi=800)

    if show_flag:
        plt.show()

    return rec_t, rec_a, target_t, target_twsa


def plot_rec_comparison(show_flag=True, save_flag=False):
    params = {
        'axes.labelsize': 20,
        'axes.titlesize': 20,
        'font.size': 20,
        'legend.fontsize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'axes.linewidth': 1.5,
        'figure.facecolor': 'w',
    }

    mpl.rcParams.update(params)
    data = np.load('data/rec_grace_gfo.npz', allow_pickle=True)
    rec_t, rec_series, _, target_t, target_series = data['all_t'], data['twsa_mean'], data['twsa_std'], data[
        'target_t'], data['target_twsa']

    # rec data
    rec_start, rec_end = '1971-01', '2021-12'
    _, _, rec_interannual, rec_seasonal, rec_resid = stl.stl_decompose(rec_series, rec_start, rec_end, plot_flag=False)
    rec_deseasoned = rec_interannual + rec_resid
    rec_t = np.arange(np.datetime64(rec_start), np.datetime64(rec_end) + 1, np.timedelta64(1, 'M'),
                      dtype='datetime64[M]')

    # calculate the WSDI
    resid = rec_deseasoned.reshape(-1, 12)  # 51 * 12
    climatology = np.mean(resid, axis=0)  # 51*12
    wsd = resid - climatology
    wsd = wsd.reshape(-1, 1)
    wsd_mean, wsd_std = np.mean(wsd), np.std(wsd)
    wsdi = (wsd - wsd_mean) / wsd_std

    # Humphrey
    hum_data = pd.read_excel('data/Humphrey.xlsx', index_col='Time')
    hum_deseasoned = np.array(hum_data['YRB'])
    hum_start, hum_end = '1971-01', '2014-12'
    hum_t = np.arange(np.datetime64(hum_start), np.datetime64(hum_end) + 1, np.timedelta64(1, 'M'),
                      dtype='datetime64[M]')

    # Li fupeng
    li_data = pd.read_excel('data/Li.xlsx', index_col='Time', sheet_name='grace_rec_deseasonalised')
    li_deseasoned = np.array(li_data['YRB'])
    li_start, li_end = '1979-07', '2020-06'
    li_t = np.arange(np.datetime64(li_start), np.datetime64(li_end) + 1, np.timedelta64(1, 'M'), dtype='datetime64[M]')

    # GLDAS
    hydro_data = pd.read_excel('data/Hydrological_Model.xlsx', index_col='Time')
    gldas = np.array(hydro_data['GLDAS_TWSA'])
    era = np.array(hydro_data['ERA_TWSA'])
    hydro_start, hydro_end = '1971-01', '2021-12'
    hydro_t = np.arange(np.datetime64(hydro_start), np.datetime64(hydro_end) + 1, np.timedelta64(1, 'M'),
                        dtype='datetime64[M]')
    _, _, _, gldas_seasonal, _ = stl.stl_decompose(gldas, hydro_start, hydro_end, plot_flag=False)
    _, _, _, era_seasonal, _ = stl.stl_decompose(era, hydro_start, hydro_end, plot_flag=False)

    # Nino34
    nino_data = pd.read_excel('data/nino34.xlsx', index_col='Time')
    nino = np.array(nino_data['Nino'])
    nino_start, nino_end = '1971-01', '2021-12'
    nino_t = np.arange(np.datetime64(nino_start), np.datetime64(nino_end) + 1, np.timedelta64(1, 'M'),
                       dtype='datetime64[M]')

    # scPDSI
    sc_pdsi_data = pd.read_excel('data/scPDSI.xlsx', index_col='Time')
    sc_pdsi = np.array(sc_pdsi_data['scPDSI'])
    sc_pdsi_start, sc_pdsi_end = '1971-01', '2018-12'
    sc_pdsi_t = np.arange(np.datetime64(sc_pdsi_start), np.datetime64(sc_pdsi_end) + 1, np.timedelta64(1, 'M'),
                          dtype='datetime64[M]')

    # TWSC and Water Budget
    twsc_wb_data = pd.read_excel('data/TWSC_WB.xlsx', index_col='Time')
    twsc = np.array(twsc_wb_data['TWSC'])
    wb = np.array(twsc_wb_data['WB'])
    twsc_wb_t = rec_t

    gs_kw = dict(height_ratios=[1., 1., 1., 1.])
    fig, axes = plt.subplots(nrows=4, ncols=1, gridspec_kw=gs_kw, layout='constrained', figsize=(24, 15))

    from matplotlib.ticker import MaxNLocator
    import matplotlib.dates as mdates

    # legend position
    num1, num2, num3, num4 = 1, 0, 3, 0
    lw = 3

    ax = axes[0]
    ax.plot(twsc_wb_t, twsc, lw=lw, label='RecNet', alpha=0.2, c='r')
    ax.plot(twsc_wb_t, wb, lw=lw, label='WB (CC=0.75)', alpha=0.2, c='b')
    twsc_smooth, wb_smooth = sf(twsc, window_length=6, polyorder=2), sf(wb, window_length=6, polyorder=2)
    ax.plot(twsc_wb_t, twsc_smooth, lw=3, label='RecNet_smoothed', c='#fd5f00', )
    ax.plot(twsc_wb_t, wb_smooth, lw=3, label='WB_smoothed (CC=0.93)', c='#0000A1')
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(MaxNLocator(24))
    ax.text(s='a', fontweight='bold', x=0.02, y=0.92,
            transform=ax.transAxes, horizontalalignment='right', fontsize=18)
    # ax.grid()
    ax.legend(bbox_to_anchor=(num1, num2), loc=num3, borderaxespad=num4)
    ax.set_ylabel('TWSC [cm]')

    ax0 = axes[-1]
    ax0.plot(rec_t, rec_deseasoned, c='r', lw=lw, label='RecNet')
    ax0.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax0.xaxis.set_major_locator(MaxNLocator(24))
    ax0.plot(hum_t, hum_deseasoned, c='b', lw=lw, label='Humphrey (CC=0.72)')
    ax0.plot(li_t, li_deseasoned, c='k', lw=lw, label='Li (CC=0.61)')
    ax0.text(s='d', fontweight='bold', x=0.02, y=0.92,
             transform=ax0.transAxes, horizontalalignment='right', fontsize=18)
    ax0.legend(bbox_to_anchor=(num1, num2), loc=num3, borderaxespad=num4)
    # ax0.grid()
    ax0.set_ylabel('Nonseasonal TWSA [cm]')

    ax1 = axes[1]
    # ax1.patch.set_facecolor('cyan')
    ax1.plot(rec_t[:372], rec_seasonal[:372], c='#FD7013', lw=lw, label='RecNet')
    ax1.plot(hydro_t[:372], gldas_seasonal[:372], c='#5e412f', lw=lw, label='GLDAS (CC=0.98)', ls='-')
    ax1.plot(hydro_t[:372], era_seasonal[:372], c='#3399FF', lw=lw, label='ERA5 (CC=0.86)', ls='-', marker='o')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_major_locator(MaxNLocator(24))
    ax1.text(s='b', fontweight='bold', x=0.02, y=0.92,
             transform=ax1.transAxes, horizontalalignment='right', fontsize=18)
    ax1.legend(bbox_to_anchor=(num1, num2), loc=num3, borderaxespad=num4)
    # ax1.grid()
    ax1.set_ylabel('Seasonal TWSA [cm]')

    ax2 = axes[2]
    nino_t, nino_data = nino_t.squeeze(), nino.squeeze()
    ax2.plot(rec_t, wsdi, label='WSDI', lw=lw, c='#097054')
    ax2.plot(sc_pdsi_t, sc_pdsi, label='scPDSI (CC=0.68)', lw=lw, c='#ef800d')
    ax2.fill_between(nino_t, nino_data, label='EI Nino', where=nino_data > 0.4, color='r', alpha=0.5)
    ax2.fill_between(nino_t, nino_data, label='La Nina', where=nino_data < -0.4, color='b', alpha=0.5)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(MaxNLocator(24))
    ax2.text(s='c', fontweight='bold', x=0.02, y=0.92,
             transform=ax2.transAxes, horizontalalignment='right', fontsize=18)
    ax2.legend(bbox_to_anchor=(num1, num2), loc=num3, borderaxespad=num4)
    ax2.set_ylabel('Index value [-]')
    # ax2.grid()

    if save_flag:
        plt.savefig('figures/Fig4.jpg', dpi=800)

    if show_flag:
        plt.show()


def plot_cross_correlation(show_flag=True, save_flag=False):
    # precipitation
    p_data = pd.read_excel('data/precip.xlsx', index_col='Time', sheet_name='2002_2017')
    p = np.array(p_data['YRB'])
    # temperature
    t_data = pd.read_excel('data/temp.xlsx', index_col='Time', sheet_name='2002_2017')
    t = np.array(t_data['YRB'])
    # GRACE
    g_data = pd.read_excel('data/GRACE.xlsx', index_col='Time')
    g = np.array(g_data['YRB'])
    # GLDAS
    # GRACE
    d_data = pd.read_excel('data/GLDAS.xlsx', index_col='Time')
    d = np.array(d_data['YRB'])

    from scipy import signal
    pg_corr = signal.correlate(p, g)
    pg_lags = signal.correlation_lags(len(p), len(g))
    pg_corr /= np.max(pg_corr)

    tg_corr = signal.correlate(t, g)
    tg_lags = signal.correlation_lags(len(t), len(g))
    tg_corr /= np.max(tg_corr)

    dg_corr = signal.correlate(d, g)
    dg_lags = signal.correlation_lags(len(d), len(g))
    dg_corr /= np.max(dg_corr)

    fig, ax = plt.subplots(3, 1, layout='constrained', figsize=(8, 6))
    idx = tuple([slice(len(p) - 12, len(p) + 12)])
    colors = ['b' if (x < max(pg_corr[idx])) else 'red' for x in pg_corr[idx]]
    ax[0].bar(pg_lags[idx], pg_corr[idx], color=colors)
    ax[0].text(s='a', fontweight='bold', x=0.98, y=0.04,
               transform=ax[0].transAxes, horizontalalignment='right', fontsize=11)

    colors = ['b' if (x < max(tg_corr[idx])) else 'red' for x in tg_corr[idx]]
    ax[1].bar(tg_lags[idx], tg_corr[idx], color=colors)
    ax[1].text(s='b', fontweight='bold', x=0.98, y=0.04,
               transform=ax[1].transAxes, horizontalalignment='right', fontsize=11)

    colors = ['b' if (x < max(dg_corr[idx])) else 'red' for x in dg_corr[idx]]
    ax[2].bar(dg_lags[idx], dg_corr[idx], color=colors)
    ax[2].text(s='c', fontweight='bold', x=0.98, y=0.04,
               transform=ax[2].transAxes, horizontalalignment='right', fontsize=11)

    fig.supylabel('correlation coefficient')
    fig.supxlabel('lag')

    if save_flag:
        plt.savefig('figures/FigS1.jpg', dpi=800)

    if show_flag:
        plt.show()


def plot_monthly_rec(show_flag=True, save_flag=False):
    fig = plt.figure(figsize=(12, 3), layout='constrained')

    gs = fig.add_gridspec(nrows=1, ncols=1)

    # reconstructed TWSA
    ax = fig.add_subplot(gs[0, :])
    rec_data = np.load('data/rec_grace_gfo.npz')
    rec, rec_std = rec_data['twsa_mean'], rec_data['twsa_std']
    ax.plot(np.arange(len(rec)), rec, c='k')
    ax.fill_between(np.arange(len(rec)), rec - 3 * rec_std, rec + 3 * rec_std, color='gray', alpha=0.5, label='Bound')
    ax.set_xticks(np.arange(0, 612, 100))
    ax.set_xticklabels(['1971-01', '1979-03', '1987-09', '1996-01', '2004-05', '2012-09', '2021-01'])
    ax.legend(loc='lower right')
    ax.set_ylabel('TWSA [cm]')

    rec_before_2000, rec_after_2000 = np.mean(rec[:348]), np.mean(rec[348:])
    ax.plot([0, 347], [rec_before_2000, rec_before_2000], color='b', ls='--')
    ax.plot([348, 611], [rec_after_2000, rec_after_2000], color='r', ls='--')

    z_b = np.polyfit(np.arange(len(rec[:348])), rec[:348], deg=1)
    poly_b = np.poly1d(z_b)
    trend_b = poly_b(np.arange(len(rec[:348])))
    ax.plot(np.arange(len(rec[:348])), trend_b, color='b', ls='-')

    z_a = np.polyfit(np.arange(len(rec[348:])), rec[348:], deg=1)
    poly_a = np.poly1d(z_a)
    trend_a = poly_a(np.arange(len(rec[348:])))
    ax.plot(np.arange(348, 612), trend_a, color='r', ls='-')

    if save_flag:
        plt.savefig('figures/FigS4.jpg', dpi=800)

    if show_flag:
        plt.show()


def plot_loss_comparison(show_flag=True, save_flag=False):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')

    data1 = np.load('results/2023-02-24 12-45-34/loss.npz', allow_pickle=True)
    tr_loss1, val_loss1 = data1['tr_loss'], data1['val_loss']  # with all possible input

    data2 = np.load('results/2023-02-26 16-47-13/loss.npz', allow_pickle=True)
    tr_loss2, val_loss2 = data2['tr_loss'], data2['val_loss']  # without dropout

    data3 = np.load('results/2023-02-26 16-50-41/loss.npz', allow_pickle=True)
    tr_loss3, val_loss3 = data3['tr_loss'], data3['val_loss']  # without dropout and with less wf

    tr_loss = [tr_loss1, tr_loss2, tr_loss3]
    val_loss = [val_loss1, val_loss2, val_loss3]

    for idx, ax in enumerate(axes.flat):
        min_tr_loss, min_val_loss = min(tr_loss[idx]), min(val_loss[idx])
        ax.plot(np.arange(len(tr_loss[idx])), tr_loss[idx], c='b', lw=2, label=f'training loss ({min_tr_loss:.2f})')
        ax.plot(np.arange(len(val_loss[idx])), val_loss[idx], c='r', lw=2,
                label=f'validation loss ({min_val_loss:.2f})')
        ax.grid()
        ax.legend()
        ax.set_xlabel('Epoch')
        if idx == 0:
            ax.set_ylabel('MSE [cm^2]')
            ax.set_title('(a) RecNet trained on the inputs listed in Table S1')
        if idx == 1:
            ax.set_title('(b) RecNet without dropout')
        if idx == 2:
            ax.set_title('(c) Simplified RecNet without dropout')

    if save_flag:
        plt.savefig('figures/FigS3.jpg', dpi=800)

    if show_flag:
        plt.show()


if __name__ == '__main__':
    print('start from here')
    save = True
    plot_cross_correlation(save_flag=save)
