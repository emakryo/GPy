import unittest
import numpy as np
from scipy.stats import norm
import GPy


class TestGPPrivPlus(unittest.TestCase):
    def test_rbf(self):
        n = 20
        d = 2
        X = np.random.randn(n, d)
        Xstar = np.random.multivariate_normal(
            np.zeros(n), (GPy.kern.RBF(d)+GPy.kern.White(d, variance=0.05)).K(X)
        ).reshape(-1, 1)

        y = (np.random.rand(n, 1) < norm.cdf(Xstar))

        m = GPy.models.GPPrivPlus(X, y, Xstar)
        print(m)

        self.assertTrue(m.checkgrad(verbose=True, tolerance=5e-3))


if __name__ == '__main__':
    unittest.main()
