

import numpy as np
from matplotlib import pyplot as plt, cm
from mpl_toolkits.mplot3d import Axes3D


def cal_expected_hft(alpha, sigma, pct_l, pct_u):
    """

    :param alpha:
    :param sigma:
    :param pct_l: such as 0.1, means that V**/V0 = 1.1
    :param pct_u: such as -0.1, means that V*/V0 = 0.9
    :return:
    """
    # if False: #sigma**2 / 2 == alpha:
    #     hft = np.log( 1+pct_l ) * np.log( (1+pct_u) ) / sigma**2
    # else:
    pow_param = 1- 2 * alpha / sigma**2
    hft = ( np.log(1.0/(1+pct_l)) - (1-np.power(1.0/(1+pct_l), pow_param))/(1-np.power((1+pct_u)/(1+pct_l), pow_param)) * np.log((1+pct_u)/(1+pct_l)) ) / (sigma ** 2 / 2 - alpha)

    idx = np.abs(sigma**2/2 - alpha) < 1e-6
    hft[idx] = np.log( 1.0/(1+pct_l[idx]) ) * np.log( (1+pct_u[idx]) ) / sigma[idx]**2
    # hft[idx] = np.log( 1.0/(1+pct_l[idx]) ) * np.log( (1+pct_u[idx]) ) / sigma**2

    return hft

def cal_linear_hft(sigma, pct_l, pct_u):
    hft = np.minimum( np.abs(np.log(1+pct_l)), np.log(1+pct_u) )
    hft = hft / (sigma / np.sqrt(365/7)) * 7

    return hft


def fht_plot():
    alpha = 0
    sigma = 1

    X = np.arange(0.79, 0.81, 0.005)  # sigma
    # X = np.arange(-1, 1, 0.01)  # alpha
    Y = np.arange(0.01, 0.2, 0.01)  # y
    # ils = il_func(X, Y)

    X, Y = np.meshgrid(X, Y)
    hft = cal_expected_hft(alpha, X, -Y, Y)
    hft = hft * 365
    hft_linear = cal_linear_hft(X, -Y, Y)

    # hft = cal_expected_hft(X, sigma, -Y, Y)
    # hft = hft * 365
    # hft_linear = cal_linear_hft(sigma, -Y, Y)


    figure = plt.figure()
    ax = Axes3D(figure)
    ax.plot_surface(X, Y, hft, rstride=1, cstride=1, cmap='rainbow')
    ax.plot_surface(X, Y, hft_linear, rstride=1, cstride=1, cmap=cm.coolwarm)
    ax.set_xlabel('sigma')
    # ax.set_xlabel('alpha')
    ax.set_ylabel('interval')
    ax.set_zlabel('First Hitting Time')
    ax.view_init(elev=30, azim=125)
    plt.show()

if __name__ == '__main__':
    fht_plot()