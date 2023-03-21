import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.io.shapereader as sr
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature
import numpy as np
import matplotlib.patches as mpatches
import xarray as xr
import geopandas as gpd
import pandas as pd
import matplotlib as mpl
from util import shp

plt.rc('font', family='sans-serif')

Yangtze_dir = r'D:\\Doctoral Documents\\program\\data\\Yangtze\\vector data\\2流域\\长江流域\\'
sub_basins_dict = {'DTH': 'Dongting Lake Basin', 'HJ': 'Hanjiang Basin', 'JLJ': 'Jialing Jiang Basin',
                   'JSJ': 'Jinsha Jiang Basin', 'MJ': 'Minjiang Basin', 'CJGL': 'Main Stream',
                   'BYH': 'Poyang Lake Basin', 'TH': 'Taihu Basin', 'WJ': 'Wujiang Basin', }

# 读取全球地形数据
ds = xr.open_dataset(r'D:\\Doctoral Documents\\program\\data\\Elevation\\ETOPO2v2c_f4.nc')
# 准备用于绘图的数据
lon = np.linspace(min(ds['x'].data), max(ds['x'].data), len(ds['x'].data))  # 经度
lat = np.linspace(min(ds['y'].data), max(ds['y'].data), len(ds['y'].data))  # 纬度
lon, lat = np.meshgrid(lon, lat)  # 构建经纬网
dem = ds['z'].data  # DEM数据


def plot_yangtze(save_plot=False, dpi=None):
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(15, 10), layout='constrained')

    gs = fig.add_gridspec(nrows=4, ncols=6, hspace=0.1, wspace=0.1)

    ax = fig.add_subplot(gs[0:2, 0:4], projection=proj)
    extent = [90, 123, 21, 38]

    # adding the DEM
    levels = np.arange(0, 8001, 100)
    cf = ax.contourf(lon, lat, dem, levels=levels, cmap='gist_rainbow')
    position = fig.add_axes([0.07, 0.54, 0.03, 0.06])  # left bottom width height
    cb = fig.colorbar(cf, cax=position, orientation='vertical', ticks=[0, 8000])
    cb.ax.set_title('DEM (m)')
    cb.ax.tick_params(labelsize=10)

    # adding the hydrological station
    ax.plot(117.67, 31, marker='^', c='green', markersize=10)  # Datong
    ax.annotate(text='DT', size=12, xy=(117.67, 31), xytext=(118, 30.5), fontweight='bold', color='green')
    ax.plot(97.24, 33.01, marker='^', c='green', markersize=10)  # Zhimenda
    ax.annotate(text='ZMD', size=12, xy=(97.24, 33.01), xytext=(96, 32), fontweight='bold', color='green')
    ax.plot(92.44, 34.22, marker='^', c='green', markersize=10)  # Tuotuohe
    ax.annotate(text='TTH', size=12, xy=(92.44, 34.22), xytext=(91.4, 33.3), fontweight='bold', color='green')
    ax.plot(115.8, 35.8, marker='^', c='green', markersize=10)  #
    ax.annotate(text='Gauging station', size=15, xy=(116.2, 36), xytext=(117, 35.5), color='k')

    # adding the main stream
    ms_shp = r'D:\Doctoral Documents\program\data\Yangtze\Changjiang\rivers.shp'
    gdf = gpd.read_file(ms_shp)
    ms_geo = gdf['geometry'][93]
    ax.add_geometries([ms_geo], crs=proj, linewidths=2, edgecolor='b',
                      facecolor='none', zorder=1)  # add the region mask
    ax.plot([91, 91], [22, 22], lw=2, c='b', label='Main stream')
    ax.legend(frameon=False, bbox_to_anchor=(0.985, 1), prop={'size': 15})

    # adding the sub-basins
    for key, value in sub_basins_dict.items():
        region_shp = Yangtze_dir + key + '.shp'

        region_mask = sr.Reader(region_shp)
        ax.add_geometries(region_mask.geometries(), crs=proj, linewidths=1, edgecolor='k',
                          facecolor='none', label=value, zorder=1)  # add the region mask

    # adding the sub-basin name
    # arrowprops = dict(arrowstyle="-", color='k')
    # size = 17
    # ax.annotate(text='Jinshajiang basin', size=size, xy=(101, 28), xytext=(94, 24), arrowprops=arrowprops)
    # ax.annotate(text='Minjiang basin', size=size, xy=(102, 31), xytext=(98, 35), arrowprops=arrowprops)
    # ax.annotate(text='Jialingjiang basin', size=size, xy=(105, 32), xytext=(102, 37), arrowprops=arrowprops)
    # ax.annotate(text='Hanjiang basin', size=size, xy=(110, 33), xytext=(108, 35.5), arrowprops=arrowprops)
    # ax.annotate(text='Taihu basin', size=size, xy=(120.5, 31), xytext=(119, 28), arrowprops=arrowprops)
    # ax.annotate(text='Poyang lake basin', size=size, xy=(115.5, 26.5), xytext=(114, 23), arrowprops=arrowprops)
    # ax.annotate(text='Dongting lake basin', size=size, xy=(111.5, 27), xytext=(107, 22), arrowprops=arrowprops)
    # ax.annotate(text='Wujiang basin', size=size, xy=(106.5, 27), xytext=(104, 24), arrowprops=arrowprops)
    # ax.annotate(text='Main stream', size=size, xy=(116, 30.5), xytext=(114, 33.5), arrowprops=arrowprops)
    # rect = mpatches.Rectangle((90.5, 20.5), width=2, height=0.8, linewidth=1, edgecolor="k", facecolor='none')
    # ax.add_artist(rect)
    # ax.text(92.8, 20.6, 'Secondary watershed boundary', fontsize=16, weight='bold')

    ax.set_extent(extent, crs=ccrs.PlateCarree())  # set the plotting extent
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))  # add the longitude
    ax.yaxis.set_major_formatter(LatitudeFormatter())  # add the latitude
    ax.set_xticks(np.arange(extent[0], extent[1] + 1, 4), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(extent[2] + 2, extent[3] + 1, 3), crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.OCEAN.with_scale('10m'), alpha=0.45, color='b')  # add the ocean
    ax.add_feature(cfeature.LAND.with_scale('10m'), alpha=0.75)  # add the land

    # ax.spines['bottom'].set_linewidth(2)
    # ax.spines['left'].set_linewidth(2)
    # ax.spines['right'].set_linewidth(2)
    # ax.spines['top'].set_linewidth(2)
    ax.tick_params(labelsize=12)
    # for tick in ax.xaxis.get_majorticklabels():
    #     tick.set_fontweight('bold')
    # for tick in ax.yaxis.get_majorticklabels():
    #     tick.set_fontweight('bold')

    # crop the DEM
    shp.shp2clip(originFig=cf, ax=ax,
                 shp_file=r'D:\\Doctoral Documents\\program\\data\\Yangtze\\yangtze vector\\长江流域范围矢量图.shp')
    add_north(ax, labelsize=16, loc_x=0.97, loc_y=0.15, width=0.025, height=0.05)

    params = {
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'font.size': 12,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'axes.linewidth': 1.,
        'figure.facecolor': 'w',
    }

    mpl.rcParams.update(params)

    # add training area
    ax_tr = fig.add_subplot(gs[2, 0:2], projection=proj)
    YRB_shp = r'D:\Doctoral Documents\program\data\Yangtze\yangtze vector\长江流域范围矢量图.shp'
    YRB_mask = sr.Reader(YRB_shp)
    ax_tr.add_geometries(YRB_mask.geometries(), crs=proj, linewidths=1, edgecolor='k',
                         facecolor='#1ee3cf')  # add the YRB mask
    extent_yrb = [81, 122, 22, 39]
    ax_tr.set_extent(extent_yrb, crs=ccrs.PlateCarree())  # set the plotting extent
    # ax_tr.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))  # add the longitude
    # ax_tr.yaxis.set_major_formatter(LatitudeFormatter())  # add the latitude
    ax_tr.set_xticks(np.arange(extent_yrb[0], extent_yrb[1] + 1, 8), crs=ccrs.PlateCarree())
    ax_tr.set_yticks(np.arange(extent_yrb[2] + 2, extent_yrb[3] + 1, 4), crs=ccrs.PlateCarree())
    ax_tr.set_facecolor('#f2f4f6')
    ax_tr.text(s='YRB', fontweight='bold', x=0.72, y=0.46,
               transform=ax_tr.transAxes, horizontalalignment='right', fontsize=12)
    ax_tr.text(s='Training Area', fontweight='light', x=0.4, y=0.15,
               transform=ax_tr.transAxes, horizontalalignment='right', fontsize=15)

    # add the precip
    ax_p = fig.add_subplot(gs[0, 4:])
    p_pd = pd.read_excel('data/precip.xlsx', sheet_name='yearly', index_col='Time')
    p = np.array(p_pd['YRB'])  # 1971-2021
    ax_p.plot(np.arange(len(p)), p, c='k')
    ax_p.set_ylabel('precipitation [cm]')
    ax_p.set_xticks(np.arange(0, 51, 10))
    ax_p.set_xticklabels(['1971', '1981', '1991', '2001', '2011', '2021'])
    p_before_2000, p_after_2000 = np.mean(p[:29]), np.mean(p[29:])
    ax_p.plot([0, 28], [p_before_2000, p_before_2000], color='b', ls='--', label='mean (1971-1999)')
    ax_p.plot([29, 50], [p_after_2000, p_after_2000], color='r', ls='--', label='mean (2000-2021)')

    z_b = np.polyfit(np.arange(len(p[:29])), p[:29], deg=1)
    poly_b = np.poly1d(z_b)
    trend_b = poly_b(np.arange(len(p[:29])))
    ax_p.plot(np.arange(len(p[:29])), trend_b, color='b', ls='-', label='trend (1971-1999)')

    z_a = np.polyfit(np.arange(len(p[29:])), p[29:], deg=1)
    poly_a = np.poly1d(z_a)
    trend_a = poly_a(np.arange(len(p[29:])))
    ax_p.plot(np.arange(29, 51), trend_a, color='r', ls='-', label='trend (2000-2021)')

    # add the temperature
    ax_t = fig.add_subplot(gs[1, 4:])
    t_pd = pd.read_excel('data/temp.xlsx', sheet_name='yearly', index_col='Time')
    t = np.array(t_pd['YRB'])
    ax_t.plot(np.arange(len(t)), t, c='k')
    ax_t.set_xticks(np.arange(0, 51, 10))
    ax_t.set_xticklabels(['1971', '1981', '1991', '2001', '2011', '2021'])
    ax_t.set_ylabel('temperature [℃]')
    t_before_2000, t_after_2000 = np.mean(t[:29]), np.mean(t[29:])
    ax_t.plot([0, 28], [t_before_2000, t_before_2000], color='b', ls='--')
    ax_t.plot([29, 50], [t_after_2000, t_after_2000], color='r', ls='--')

    z_b = np.polyfit(np.arange(len(t[:29])), t[:29], deg=1)
    poly_b = np.poly1d(z_b)
    trend_b = poly_b(np.arange(len(t[:29])))
    ax_t.plot(np.arange(len(t[:29])), trend_b, color='b', ls='-')

    z_a = np.polyfit(np.arange(len(t[29:])), t[29:], deg=1)
    poly_a = np.poly1d(z_a)
    trend_a = poly_a(np.arange(len(t[29:])))
    ax_t.plot(np.arange(29, 51), trend_a, color='r', ls='-')

    # add the reconstruction
    ax_rec = fig.add_subplot(gs[2, 4:])
    rec_pd = pd.read_excel('data/reconstruction.xlsx', sheet_name='yearly', index_col='Time')
    rec = np.array(rec_pd['YRB'])
    rec_std = np.array(rec_pd['STD'])
    ax_rec.fill_between(np.arange(len(rec)), rec - 3 * rec_std, rec + 3 * rec_std, color='#FD7013', alpha=0.4,
                        label='Bound')
    ax_rec.plot(np.arange(len(rec)), rec, c='k')
    ax_rec.set_xticks(np.arange(0, 51, 10))
    ax_rec.set_xticklabels(['1971', '1981', '1991', '2001', '2011', '2021'])
    ax_rec.set_ylabel('TWSA [cm]')
    rec_before_2000, rec_after_2000 = np.mean(rec[:29]), np.mean(rec[29:])
    ax_rec.plot([0, 28], [rec_before_2000, rec_before_2000], color='b', ls='--', label='mean (1971-1999)')
    ax_rec.plot([29, 50], [rec_after_2000, rec_after_2000], color='r', ls='--', label='mean (2000-2021)')

    z_b = np.polyfit(np.arange(len(rec[:29])), rec[:29], deg=1)
    poly_b = np.poly1d(z_b)
    trend_b = poly_b(np.arange(len(rec[:29])))
    ax_rec.plot(np.arange(len(rec[:29])), trend_b, color='b', ls='-', label='trend (1971-1999)')

    z_a = np.polyfit(np.arange(len(rec[29:])), rec[29:], deg=1)
    poly_a = np.poly1d(z_a)
    trend_a = poly_a(np.arange(len(rec[29:])))
    ax_rec.plot(np.arange(29, 51), trend_a, color='r', ls='-', label='trend (2000-2021)')

    # add the tuotuohe
    ax_tuo = fig.add_subplot(gs[3, 0:2])
    tuo_pd = pd.read_excel('data/runoff.xlsx', sheet_name='沱沱河', index_col='年')
    tuo = np.array(tuo_pd['EWH_cm'])[7:]  # from 1971-2019
    ax_tuo.plot(np.arange(len(tuo)), tuo, c='k')
    ax_tuo.set_xticks(np.arange(0, 51, 10))
    ax_tuo.set_xticklabels(['1971', '1981', '1991', '2001', '2011', '2021'])
    ax_tuo.set_ylabel('runoff for TTH [cm]')
    tuo_before_2000, tuo_after_2000 = np.mean(tuo[:29]), np.mean(tuo[29:])
    ax_tuo.plot([0, 28], [tuo_before_2000, tuo_before_2000], color='b', ls='--')
    ax_tuo.plot([29, 48], [tuo_after_2000, tuo_after_2000], color='r', ls='--')

    z_b = np.polyfit(np.arange(len(tuo[:29])), tuo[:29], deg=1)
    poly_b = np.poly1d(z_b)
    trend_b = poly_b(np.arange(len(tuo[:29])))
    ax_tuo.plot(np.arange(len(tuo[:29])), trend_b, color='b', ls='-')

    z_a = np.polyfit(np.arange(len(tuo[29:])), tuo[29:], deg=1)
    poly_a = np.poly1d(z_a)
    trend_a = poly_a(np.arange(len(tuo[29:])))
    ax_tuo.plot(np.arange(29, 49), trend_a, color='r', ls='-')

    # add the zhimenda
    ax_zmd = fig.add_subplot(gs[3, 2:4])
    zmd_pd = pd.read_excel('data/runoff.xlsx', sheet_name='直门达', index_col='年')
    zmd = np.array(zmd_pd['EWH_cm'])[7:]  # from 1971
    ax_zmd.plot(np.arange(len(zmd)), zmd, c='k')
    ax_zmd.set_xticks(np.arange(0, 51, 10))
    ax_zmd.set_xticklabels(['1971', '1981', '1991', '2001', '2011', '2021'])
    ax_zmd.set_ylabel('runoff for ZMD [cm]')
    zmd_before_2000, zmd_after_2000 = np.mean(zmd[:29]), np.mean(zmd[29:])
    ax_zmd.plot([0, 28], [zmd_before_2000, zmd_before_2000], color='b', ls='--')
    ax_zmd.plot([29, 50], [zmd_after_2000, zmd_after_2000], color='r', ls='--')

    z_b = np.polyfit(np.arange(len(zmd[:29])), zmd[:29], deg=1)
    poly_b = np.poly1d(z_b)
    trend_b = poly_b(np.arange(len(zmd[:29])))
    ax_zmd.plot(np.arange(len(zmd[:29])), trend_b, color='b', ls='-')

    z_a = np.polyfit(np.arange(len(zmd[29:])), zmd[29:], deg=1)
    poly_a = np.poly1d(z_a)
    trend_a = poly_a(np.arange(len(zmd[29:])))
    ax_zmd.plot(np.arange(29, 51), trend_a, color='r', ls='-')

    # add the DT
    ax_dt = fig.add_subplot(gs[3, 4:])
    dt_pd = pd.read_excel('data/runoff.xlsx', sheet_name='大通站', index_col='年')
    dt = np.array(dt_pd['EWH_cm'])[11:]  # from 1971
    ax_dt.plot(np.arange(len(dt)), dt, c='k')
    ax_dt.set_xticks(np.arange(0, 51, 10))
    ax_dt.set_xticklabels(['1971', '1981', '1991', '2001', '2011', '2021'])
    ax_dt.set_ylabel('runoff for DT [cm]')

    dt_before_2000, dt_after_2000 = np.mean(dt[:29]), np.mean(dt[29:])
    ax_dt.plot([0, 28], [dt_before_2000, dt_before_2000], color='b', ls='--', label='mean (1971-1999)')
    ax_dt.plot([29, 50], [dt_after_2000, dt_after_2000], color='r', ls='--', label='mean (2000-2021)')

    z_b = np.polyfit(np.arange(len(dt[:29])), dt[:29], deg=1)
    poly_b = np.poly1d(z_b)
    trend_b = poly_b(np.arange(len(dt[:29])))
    ax_dt.plot(np.arange(len(dt[:29])), trend_b, color='b', ls='-', label='trend (1971-1999)')

    z_a = np.polyfit(np.arange(len(dt[29:])), dt[29:], deg=1)
    poly_a = np.poly1d(z_a)
    trend_a = poly_a(np.arange(len(dt[29:])))
    ax_dt.plot(np.arange(29, 51), trend_a, color='r', ls='-', label='trend (2000-2021)')

    lines, labels = fig.axes[5].get_legend_handles_labels()
    fig.legend(lines, labels, bbox_to_anchor=(0.62, 0.47), framealpha=1, prop={'size': 16})

    x, y = 0.04, 0.90
    fs = 12
    ax.text(s='a', fontweight='bold', x=0.02, y=0.96,
            transform=ax.transAxes, horizontalalignment='right', fontsize=fs)
    ax_tr.text(s='b', fontweight='bold', x=x, y=y,
               transform=ax_tr.transAxes, horizontalalignment='right', fontsize=fs)
    ax_tuo.text(s='f', fontweight='bold', x=x, y=y,
                transform=ax_tuo.transAxes, horizontalalignment='right', fontsize=fs)
    ax_zmd.text(s='g', fontweight='bold', x=x, y=y,
                transform=ax_zmd.transAxes, horizontalalignment='right', fontsize=fs)
    ax_dt.text(s='h', fontweight='bold', x=x, y=y,
               transform=ax_dt.transAxes, horizontalalignment='right', fontsize=fs)
    ax_p.text(s='c', fontweight='bold', x=x, y=y,
              transform=ax_p.transAxes, horizontalalignment='right', fontsize=fs)
    ax_t.text(s='d', fontweight='bold', x=x, y=y,
              transform=ax_t.transAxes, horizontalalignment='right', fontsize=fs)
    ax_rec.text(s='e', fontweight='bold', x=x, y=y,
                transform=ax_rec.transAxes, horizontalalignment='right', fontsize=fs)

    if save_plot:
        plt.savefig('figures/Fig1.jpg', dpi=dpi)

    plt.show()


def add_north(ax, labelsize=18, loc_x=0.88, loc_y=0.85, width=0.06, height=0.09, pad=0.14):
    """
    画一个比例尺带'N'文字注释
    主要参数如下
    :param ax: 要画的坐标区域 Axes实例 plt.gca()获取即可
    :param labelsize: 显示'N'文字的大小
    :param loc_x: 以文字下部为中心的占整个ax横向比例
    :param loc_y: 以文字下部为中心的占整个ax纵向比例
    :param width: 指南针占ax比例宽度
    :param height: 指南针占ax比例高度
    :param pad: 文字符号占ax比例间隙
    :return: None
    """
    minx, maxx = ax.get_xlim()
    miny, maxy = ax.get_ylim()
    ylen = maxy - miny
    xlen = maxx - minx
    left = [minx + xlen * (loc_x - width * .5), miny + ylen * (loc_y - pad)]
    right = [minx + xlen * (loc_x + width * .5), miny + ylen * (loc_y - pad)]
    top = [minx + xlen * loc_x, miny + ylen * (loc_y - pad + height)]
    center = [minx + xlen * loc_x, left[1] + (top[1] - left[1]) * .4]
    triangle = mpatches.Polygon([left, top, right, center], color='k')
    ax.text(s='N',
            x=minx + xlen * loc_x,
            y=miny + ylen * (loc_y - pad + height),
            fontsize=labelsize,
            horizontalalignment='center',
            verticalalignment='bottom')
    ax.add_patch(triangle)


if __name__ == '__main__':
    print('starting...')
    plot_yangtze(save_plot=True, dpi=800)
