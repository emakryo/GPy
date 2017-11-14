import unittest
import numpy as np
import GPy

class TestPrivTransferModels(unittest.TestCase):
    def setUp(self):
        random_state = 0
        generator = np.random.RandomState(random_state)
        self.generator = generator

        self.dim = 1
        self.n_samples = 10
        self.X = generator.rand(self.n_samples, self.dim) * 10
        self.S = generator.multivariate_normal(
            np.zeros(self.n_samples),
            GPy.kern.RBF(self.dim).K(self.X) + 0.05 * np.eye(self.n_samples)
        ).reshape(-1, 1)
        # self.S = -self.X - 0.5
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

    def testGPPrivTransfer(self):
        m = GPy.models.GPPrivTransfer(self.X, self.Y, self.S)
        m.randomize(self.generator.normal)
        print(m)
        assert m.checkgrad(verbose=True, tolerance=1e-2)
        print(m)
        assert m.checkgrad(verbose=True, tolerance=1e-2)
        # assert m.log_likelihood() < 0
        fm, fv = m.predict_noiseless(self.Xgrid_1d)


    def testConditionalEP(self):
        m0 = GPy.models.GPPrivTransfer(self.X, self.Y, self.S)
        m1 = GPy.models.GPPrivTransfer(self.X, self.Y, self.S)
        m1.inference_method = GPy.inference.latent_function_inference.EP(ep_mode='nested')
        m1.parameters_changed()
        m2 = GPy.models.GPRegression(self.X, self.S)

        print(m0)
        print(m1)
        print(m2)

        assert np.allclose(m0.log_likelihood(), m1.log_likelihood() - m2.log_likelihood())
        assert m0.checkgrad(verbose=True)
        assert m1.checkgrad(verbose=True)
        assert m2.checkgrad(verbose=True)

        dL_dK = m1.grad_dict['dL_dK']
        dL_dK[self.n_samples:, self.n_samples:] -= m2.grad_dict['dL_dK']
        assert np.allclose(m0.grad_dict['dL_dK'], dL_dK)


    def testGPPrivTransfer_linear(self):
        m = GPy.models.GPPrivTransfer(self.X, self.Y, self.S, kernel=self.linear)
        m.randomize(self.generator.normal)
        print(m)
        assert m.checkgrad(verbose=True)
        m.optimize()
        assert m.checkgrad(verbose=True)
        # assert m.log_likelihood() < 0
        fm, fv = m.predict_noiseless(self.Xgrid_1d)

    def testKernel(self):
        X, Y, output_index = GPy.util.multioutput.build_XY([self.X, self.X+0.*np.random.randn(self.n_samples, self.dim)],
                                                           [self.S, self.S])
        kernel = GPy.kern.RBF(self.dim).prod(GPy.kern.DualTask(1, active_dims=[self.dim]))
        likelihood = GPy.likelihoods.GaussianPV()
        # likelihood = GPy.likelihoods.Bernoulli()
        m = GPy.core.GP(X, Y, kernel=kernel, likelihood=likelihood,
                        inference_method=GPy.inference.latent_function_inference.EP(ep_mode='nested', max_iters=500)
                        )
        m.randomize(self.generator.normal)
        print(m)
        assert m.checkgrad(verbose=True)
        m.optimize()
        print(m)
        assert m.checkgrad(verbose=True)
        # assert m.log_likelihood() < 0


if __name__ == "__main__":
    unittest.main()
