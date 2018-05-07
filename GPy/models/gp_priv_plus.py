import warnings
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from .gp_priv_base import GPPrivBase
from ..likelihoods.link_functions import Heaviside
from ..models.gp_priv_plus_inner_calc import (_Z_cy, _dZdm_cy, _d2Zdm2_cy,
                                              _dZdm_star_cy, _d2Zdm_star2_cy)


use_cython = True
epsilon = np.finfo(np.float64).eps


class NegativeVarianceWarning(Warning):
    pass


def _Z_py(g, mean, var, mean_star, var_star, y):
    return (norm.cdf(y * mean / np.sqrt(var + np.exp(g))) *
            norm.pdf(g, mean_star, np.sqrt(var_star)))


def _dZdm_py(g, mean, var, mean_star, var_star, y):
    return (y / np.sqrt(var + np.exp(g)) *
            norm.pdf(y * mean / np.sqrt(var + np.exp(g))) *
            norm.pdf(g, mean_star, np.sqrt(var_star)))


def _d2Zdm2_py(g, mean, var, mean_star, var_star, y):
    return (-y * mean / np.sqrt(var + np.exp(g)) ** 3 *
            norm.pdf(y * mean / np.sqrt(var + np.exp(g))) *
            norm.pdf(g, mean_star, np.sqrt(var_star)))


def _dZdm_star_py(g, mean, var, mean_star, var_star, y):
    return (norm.cdf(y * mean / np.sqrt(var + np.exp(g))) *
            (g - mean_star) / var_star *
            norm.pdf(g, mean_star, np.sqrt(var_star)))


def _d2Zdm_star2_py(g, mean, var, mean_star, var_star, y):
    return (norm.cdf(y * mean / np.sqrt(var + np.exp(g))) *
            ((g - mean_star) ** 2 / var_star - 1) / var_star *
            norm.pdf(g, mean_star, np.sqrt(var_star)))


if use_cython:
    _Z = _Z_cy
    _dZdm = _dZdm_cy
    _d2Zdm2 = _d2Zdm2_cy
    _dZdm_star = _dZdm_star_cy
    _d2Zdm_star2 = _d2Zdm_star2_cy
else:
    _Z = _Z_py
    _dZdm = _dZdm_py
    _d2Zdm2 = _d2Zdm2_py
    _dZdm_star = _dZdm_star_py
    _d2Zdm_star2 = _d2Zdm_star2_py


class GPPrivPlus(GPPrivBase):
    def __init__(self, X, Y, Xstar, kernel=None, kernel_star=None,
                 mean=None, mean_star=None, max_iter=100,
                 damping=0.9, init_damping=0.,
                 parallel_update=True, ignore_warnings=True, show_progress=False,
                 name="gp_priv_plus"):
        super(GPPrivPlus, self).__init__(
                X, Y, Xstar, kernel, kernel_star, mean, mean_star, max_iter,
                damping, init_damping, parallel_update, show_progress, name)

        if ignore_warnings:
            warnings.simplefilter('ignore', NegativeVarianceWarning)

    @classmethod
    def _next_site(cls, cav, cav_star, y):
        Z = quad(_Z, -np.inf, np.inf,
                 (cav.mean, cav.var, cav_star.mean, cav_star.var, y))[0]
        dZdm = quad(_dZdm, -np.inf, np.inf,
                    (cav.mean, cav.var, cav_star.mean, cav_star.var, y))[0]
        d2Zdm2 = quad(_d2Zdm2, -np.inf, np.inf,
                      (cav.mean, cav.var, cav_star.mean, cav_star.var, y))[0]
        dZdm_star = quad(_dZdm_star, -np.inf, np.inf,
                         (cav.mean, cav.var, cav_star.mean, cav_star.var, y))[0]
        d2Zdm_star2 = quad(_d2Zdm_star2, -np.inf, np.inf,
                           (cav.mean, cav.var, cav_star.mean, cav_star.var, y))[0]
        nu, tau = _next_site_common(Z, dZdm, d2Zdm2, cav)
        nu_star, tau_star = _next_site_common(Z, dZdm_star, d2Zdm_star2, cav_star)
        if tau < epsilon:
            warnings.warn("Tau of site distribution %f < 0" % tau,
                          NegativeVarianceWarning)
            tau = epsilon

        if tau_star < epsilon:
            warnings.warn("Tau of noise site distribution %f < 0" % tau_star,
                          NegativeVarianceWarning)
            tau_star = epsilon

        return nu, tau, nu_star, tau_star, np.log(Z)


def _next_site_common(Z, dZdm, d2Zdm2, cav):
    alpha = dZdm / Z
    beta = d2Zdm2 / Z - alpha**2
    nu = (alpha - cav.mean * beta) / (1 + cav.var * beta)
    tau = - beta / (1 + cav.var * beta)
    # z = (np.exp(-0.5 * cav.mean * alpha**2 / (2 + cav.mean * beta)) /
    #      np.sqrt(2*np.pi*cav.var*(2+cav.var*beta)))
    return nu, tau
