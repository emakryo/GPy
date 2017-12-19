# cython: profile=True
from libc.math cimport exp, sqrt, erf, M_PI

DEF sq2 = 1.4142135623730951
DEF sq2pi = 2.5066282746310002

cdef double pdf(double x, double mean, double std):
    return exp(-(x - mean) * (x - mean) / (2 * std * std)) / (sq2pi * std)


cdef double cdf(double z):
    return 0.5 * (1 + erf(z / sq2))


def _Z_cy(double g, double mean, double var,
                  double mean_star, double var_star, double y):
    return (cdf(y * mean / sqrt(var + exp(g))) *
            pdf(g, mean_star, sqrt(var_star)))


def _dZdm_cy(double g, double mean, double var,
                     double mean_star, double var_star, double y):
    cdef double sqv = sqrt(var + exp(g))
    return (y / sqv * pdf(y * mean / sqv, 0, 1) *
            pdf(g, mean_star, sqrt(var_star)))


def _d2Zdm2_cy(double g, double mean, double var,
                       double mean_star, double var_star, double y):
    cdef double sqv = sqrt(var + exp(g))
    return (-y * mean / sqv ** 3 *
            pdf(y * mean / sqv, 0, 1) *
            pdf(g, mean_star, sqrt(var_star)))


def _dZdm_star_cy(double g, double mean, double var,
                       double mean_star, double var_star, double y):
    return (cdf(y * mean / sqrt(var + exp(g))) *
            (g - mean_star) / var_star *
            pdf(g, mean_star, sqrt(var_star)))


def _d2Zdm_star2_cy(double g, double mean, double var,
                       double mean_star, double var_star, double y):
    return (cdf(y * mean / sqrt(var + exp(g))) *
            ((g - mean_star) ** 2 / var_star - 1) / var_star *
            pdf(g, mean_star, sqrt(var_star)))
