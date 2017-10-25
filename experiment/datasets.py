import numpy as np
import GPy

def test1(n_data=200, dim=1):
    x = np.random.rand(n_data, dim) * 10
    s = np.random.multivariate_normal(np.zeros(n_data),
                                      GPy.kern.RBF(dim).K(x) +
                                      0.05 * np.eye(n_data)).reshape(-1, 1)
    y = np.sign(s)
    return x, y, s


def toy1(n_data=200, dim=50):
    alpha = np.random.randn(dim, 1)
    x = np.random.randn(n_data, dim)
    s = x @ alpha
    eps = np.random.randn(n_data, 1)
    y = np.sign(s+eps)
    return x, y, s

def toy2(n_data=200, dim=50):
    alpha = np.random.randn(dim, 1)
    s = np.random.randn(n_data, dim)
    x = s + np.random.randn(n_data, dim)
    y = np.sign(s @ alpha)
    return x, y, s

def toy3(n_data=200, dim=50, n_active=3):
    alpha = np.random.randn(n_active, 1)
    active_dims = np.random.permutation(dim)[:n_active]
    x = np.random.randn(n_data, dim)
    s = x[:, active_dims]
    y = np.sign(s @ alpha)
    return x, y, s

def toy4(n_data=200, dim=50, n_active=3):
    alpha = np.random.randn(n_active, 1)
    x = np.random.randn(n_data, dim)
    active_dims = np.array([np.random.permutation(dim)[:n_active] for _ in range(n_data)])
    s = np.array([xi[j] for xi, j in zip(x, active_dims)])
    y = np.sign(s @ alpha)
    return x, y, s
