import unittest
import numpy as np, GPy
from nose.tools import with_setup

class TestDistilltionModels(unittest.TestCase):
    def setUp(self):
        random_state = 0
        generator = np.random.RandomState(random_state)

        self.n_samples = 50
        self.X = generator.rand(self.n_samples, 1) * 10
        self.S = generator.multivariate_normal(np.zeros(self.n_samples),
                GPy.kern.RBF(1).K(self.X) + 0.05 * np.eye(self.n_samples)).reshape(-1, 1)
        self.Y = np.sign(self.S)

        self.n_grid = 50
        self.Xgrid = np.zeros((self.n_grid, 2))
        self.Xgrid[:, 0] = np.linspace(0, 10, self.n_grid)

    def tearDown(self):
        self.n_samples = None
        self.X = None
        self.S = None
        self.Y = None
        self.n_grid = None
        self.Xgrid = None

    @with_setup(setUp, tearDown)
    def testGPDistillation(self):
        m = GPy.models.GPDistillation(self.X, self.Y, self.S)
        m.optimize()
        fm, fv = m.predict_noiseless(self.Xgrid)

if __name__ == "__main__":
    unittest.main()
