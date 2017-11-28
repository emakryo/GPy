import unittest
import numpy as np
from scipy.stats import norm
import GPy


class TestGPPrivPlus(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_rbf(self):
        n_all = 30
        d = 1
        X = np.random.randn(n_all, d)
        Xstar = np.random.multivariate_normal(
            np.zeros(n_all), (GPy.kern.RBF(d)+GPy.kern.White(d, variance=0.05)).K(X)
        ).reshape(-1, 1)
        y = (np.random.rand(n_all, 1) < norm.cdf(Xstar))

        n_train = 20
        Xtr = X[:n_train]
        Xtr_star = Xstar[:n_train]
        ytr = y[:n_train]
        Xte = X[n_train:]
        yte = y[n_train:]

        m = GPy.models.GPPrivPlus(Xtr, ytr, Xtr_star)
        print(m)
        self.assertTrue(m.checkgrad(verbose=True, tolerance=3e-2))

        m.optimize(messages=True)
        print(m)
        accuracy = np.count_nonzero((m.predict(Xte)[0]-0.5) * yte > 0) / (n_all-n_train)
        print(accuracy)


if __name__ == '__main__':
    #unittest.main()
    TestGPPrivPlus().test_rbf()
