import numpy as np

epsilon = 0.00001


def corr(x, y):
    """
    calculate the correlation coefficient
    """
    dim_x = len(x.shape)
    dim_y = len(y.shape)
    assert dim_x == dim_y
    if dim_x <= 2:
        x_mean, y_mean = np.mean(x), np.mean(y)
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sqrt(np.sum((x - x_mean) ** 2)) * np.sqrt(np.sum((y - y_mean) ** 2))
        return numerator / (denominator + epsilon)
    elif dim_x == 3:  # calculate the spatial corr
        x_mean, y_mean = np.mean(x, axis=0), np.mean(y, axis=0)
        numerator = np.sum((x - x_mean) * (y - y_mean), axis=0)
        denominator = np.sqrt(np.sum((x - x_mean) ** 2, axis=0)) * np.sqrt(np.sum((y - y_mean) ** 2, axis=0))
        return numerator / (denominator + epsilon)


def mae(x, y):
    """calculate the mean absolute error"""
    dim_x = len(x.shape)
    dim_y = len(y.shape)
    assert dim_x == dim_y
    if dim_x <= 2:
        return np.mean(np.abs(y - x))
    elif dim_x == 3:
        return np.mean(np.abs(y - x), axis=0)


def rmse(x, y):
    """Root Mean Square Error"""
    dim_x = len(x.shape)
    dim_y = len(y.shape)
    assert dim_x == dim_y
    if dim_x <= 2:
        return np.sqrt(np.mean((y - x) ** 2))
    elif dim_x == 3:
        return np.sqrt(np.mean((y - x) ** 2, axis=0))


def nrmse(obs, pre):
    """Normalized Root Mean Square Error"""
    dim_obs = len(obs.shape)
    dim_pre = len(pre.shape)
    assert dim_obs == dim_pre
    if dim_obs <= 2:
        return rmse(obs, pre) / (np.max(obs) - np.min(obs) + epsilon)
    elif dim_obs == 3:
        return rmse(obs, pre) / (np.max(obs, axis=0) - np.min(obs, axis=0) + epsilon)


def nse(prediction, target):
    """Nash-Sutcliffe Efficiency, NSE"""

    dim_p = len(prediction.shape)
    dim_t = len(target.shape)
    assert dim_t == dim_p
    if dim_t <= 2:
        target_mean = np.mean(target)
        numerator = np.sum((target - prediction) ** 2)
        denominator = np.sum((target - target_mean) ** 2)
        return 1 - numerator / (denominator + epsilon)
    elif dim_t == 3:
        target_mean = np.mean(target, axis=0)
        numerator = np.sum((target - prediction) ** 2, axis=0)
        denominator = np.sum((target - target_mean) ** 2, axis=0)
        return 1 - numerator / (denominator + epsilon)


if __name__ == '__main__':
    print('start...')
