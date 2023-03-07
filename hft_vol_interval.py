# Material
#   http://marcoagd.usuarios.rdc.puc-rio.br/hittingt.html#spreadsheet
#   Arithmetic Brownian Motion: dV = alpha * dt + sigma * dZ
#   Geometric Brownian Motion: dv = d(lnV) = (alpha - ½ sigma^2) * dt + sigma * dZ
#       Following Dixit's textbook The Art of Smooth Pasting (p.7):
#       dV/V = alpha * dt + sigma * dZ, letting v = ln(V), and using Itô's Lemma we find that v follows the above arithmetic (or ordinary) Brownian motion
#       Although the volatility term is the same, as highlighted by Dixit, d(lnV) is different of dV/V - in reality, by the Jensen's inequality, d(lnV) < dV/V.
#       There is a frequent confusion between d(lnV) and dV/V (people saying that is the same thing).
#       Sometimes this confusion has no practical importance because the drift value doesn't matter for several applications
#       (the case of many options calculations, like the Black & Scholes famous equation), but this is not the case here.
#       As discussed before, for the hitting time calculations, the drift a matters, and there is a difference of ½ * sigma^2 dt between these processes.

import numpy as np
from matplotlib import pyplot as plt, cm
from mpl_toolkits.mplot3d import Axes3D


def cal_expected_hft(alpha, sigma, pct_l, pct_u):
    """
    :param alpha: drift, p.a.
    :param sigma: volatility, p.a.
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
    if isinstance(idx, np.ndarray):
        sigma_idx = sigma[idx] if isinstance(sigma, np.ndarray) else sigma
        pct_l_idx = pct_l[idx] if isinstance(pct_l, np.ndarray) else pct_l
        pct_u_idx = pct_u[idx] if isinstance(pct_u, np.ndarray) else pct_u
        hft[idx] = np.log( 1.0/(1+pct_l_idx) ) * np.log( (1+pct_u_idx) ) / sigma_idx**2
    else:
        if idx:
            hft = np.log(1.0 / (1 + pct_l)) * np.log((1 + pct_u)) / sigma ** 2

    return hft

def cal_linear_hft(sigma, pct_l, pct_u):
    hft = np.minimum( np.abs(np.log(1+pct_l)), np.log(1+pct_u) )
    hft = hft / (sigma / np.sqrt(365/7)) * 7

    return hft


def fht_plot_alpha_interval():
    sigma = 1

    X = np.arange(-1, 1, 0.01)  # alpha
    Y = np.arange(0.01, 0.2, 0.01)  # y

    X, Y = np.meshgrid(X, Y)

    hft = cal_expected_hft(X, sigma, -Y, Y)
    hft = hft * 365
    hft_linear = cal_linear_hft(sigma, -Y, Y)

    figure = plt.figure()
    ax = Axes3D(figure)
    ax.plot_surface(X, Y, hft, rstride=1, cstride=1, cmap='rainbow')
    ax.plot_surface(X, Y, hft_linear, rstride=1, cstride=1, cmap=cm.coolwarm)
    ax.set_xlabel('alpha')
    ax.set_ylabel('interval')
    ax.set_zlabel('First Hitting Time')
    ax.view_init(elev=30, azim=125)
    plt.show()

def fht_plot_sigma_interval():
    alpha = 0.5

    # X = np.arange(0.79, 0.81, 0.005)  # sigma
    X = np.arange(0.5, 1.5, 0.01)  # sigma
    Y = np.arange(0.1, 0.4, 0.01)  # y
    X, Y = np.meshgrid(X, Y)

    hft = cal_expected_hft(alpha, X, -Y, Y)
    hft = hft * 365
    hft_linear = cal_linear_hft(X, -Y, Y)

    figure = plt.figure()
    ax = Axes3D(figure)
    ax.plot_surface(X, Y, hft, rstride=1, cstride=1, cmap='rainbow')
    ax.plot_surface(X, Y, hft_linear, rstride=1, cstride=1, cmap=cm.coolwarm)
    ax.set_xlabel('sigma')
    ax.set_ylabel('interval')
    ax.set_zlabel('First Hitting Time')
    ax.view_init(elev=30, azim=125)
    plt.show()

if __name__ == '__main__':
    # fht_plot_sigma_interval()
    # fht_plot_alpha_interval()
    hft = cal_expected_hft(0.5, 1, -0.3, 0.3)
    print('Done')