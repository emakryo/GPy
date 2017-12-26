import unittest
import numpy as np
from scipy.stats import norm
import GPy

class TestPrivTransferModels(unittest.TestCase):
    def setUp(self):
        self.rs = np.random.RandomState(0)

    def gen_data(self, n_data, dim):
        x = 10 * self.rs.rand(n_data, dim)
        xs = self.rs.multivariate_normal(
            np.zeros(n_data),
            GPy.kern.RBF(dim).K(x)
        )[:, None]
        y = np.where(self.rs.rand(n_data, 1) < norm.cdf(xs), 1, -1)

        m = GPy.models.GPClassification(xs, y)
        f, v = m.predict_noiseless(xs)

        return x, y, f, v

    def testGPPrivTransfer(self):
        n_tr = 20
        n_te = 100
        dim = 1
        x, y, f, v = self.gen_data(n_tr+n_te, dim)
        x_tr, x_te = x[:n_tr], x[n_tr:]
        y_tr, y_te = y[:n_tr], y[n_tr:]
        f_tr, f_te = f[:n_tr], f[n_tr:]
        v_tr, v_te = v[:n_tr], v[n_tr:]
        m = GPy.models.GPPrivTransfer(x_tr, y_tr, f_tr, v_tr)
        m.randomize(self.rs.normal)
        print(m)
        assert m.checkgrad(verbose=True)
        print(m)
        assert m.checkgrad(verbose=True)
        # assert m.log_likelihood() < 0

        f_pr, v_pt = m.predict_noiseless(x_te)

        print(np.count_nonzero(f_pr * y_te > 0) / n_te)


    def testConditionalEP(self):
        n = 20
        x, y, f, v = self.gen_data(n, 1)
        m0 = GPy.models.GPPrivTransfer(x, y, f, v)
        m1 = GPy.models.GPPrivTransfer(x, y, f, v)
        m1.inference_method = GPy.inference.latent_function_inference.EP(ep_mode='nested')
        m1.parameters_changed()
        m2 = GPy.core.GP(x, f, kernel=m0.base_kernel.copy(),
                         likelihood=m0.likelihood_list[1].copy())

        print(m0)
        print(m1)
        print(m2)

        assert np.allclose(m0.log_likelihood(), m1.log_likelihood() - m2.log_likelihood())
        assert m0.checkgrad(verbose=True)
        assert m1.checkgrad(verbose=True)
        assert m2.checkgrad(verbose=True)

        dL_dK = m1.grad_dict['dL_dK']
        dL_dK[n:, n:] -= m2.grad_dict['dL_dK']
        assert np.allclose(m0.grad_dict['dL_dK'], dL_dK)


    def testGPPrivTransfer_linear(self):
        n = 20
        dim = 2
        x, y, f, v = self.gen_data(n, dim)
        m = GPy.models.GPPrivTransfer(x, y, f, v, kernel=GPy.kern.Linear(dim))
        m.randomize(self.rs.normal)
        print(m)
        assert m.checkgrad(verbose=True)
        m.optimize()
        assert m.checkgrad(verbose=True)
        # assert m.log_likelihood() < 0
        fm, fv = m.predict_noiseless(x)


if __name__ == "__main__":
    unittest.main()
