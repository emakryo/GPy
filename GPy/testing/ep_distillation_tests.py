import unittest
import numpy as np
import GPy

class TestDistilltionModels(unittest.TestCase):
    def setUp(self):
        random_state = 0
        generator = np.random.RandomState(random_state)

        self.dim = 1
        self.n_samples = 50
        self.X = generator.rand(self.n_samples, self.dim) * 10
        self.S = generator.multivariate_normal(np.zeros(self.n_samples),
                GPy.kern.RBF(self.dim).K(self.X) + 0.05 * np.eye(self.n_samples)).reshape(-1, 1)
        self.Y = np.sign(self.S)

        self.n_grid = 50
        self.Xgrid_1d = np.zeros((self.n_grid, 2))
        self.Xgrid_1d[:, 0] = np.linspace(0, 10, self.n_grid)

        self.linear = GPy.kern.Linear(self.dim)

    def tearDown(self):
        self.n_samples = None
        self.X = None
        self.S = None
        self.Y = None
        self.n_grid = None
        self.Xgrid_1d = None

    def testGPDistillation(self):
        m = GPy.models.GPDistillation(self.X, self.Y, self.S)
        m.randomize()
        print(m)
        assert m.checkgrad(verbose=True)
        fm, fv = m.predict_noiseless(self.Xgrid_1d)

    def testGPDistillation_linear(self):
        m = GPy.models.GPDistillation(self.X, self.Y, self.S, kernel=self.linear)
        m.randomize()
        print(m)
        assert m.checkgrad(verbose=True)
        fm, fv = m.predict_noiseless(self.Xgrid_1d)


if __name__ == "__main__":
    unittest.main()
