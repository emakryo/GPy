import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import GPy
import datasets


def toy1(n_train=200, n_test=10000, dim=50, rep=100, verbose=False):
    dist_result = []
    plain_result = []
    priv_result = []
    for _ in range(rep):
        x, y, s = datasets.toy1(n_train+n_test, dim=dim)
        x_train = x[:n_train]
        x_test = x[n_train:]
        y_train = y[:n_train]
        y_test = y[n_train:]
        s_train = s[:n_train]
        s_test = s[n_train:]

        # plt.plot(x_train, y_train, "o")
        # plt.plot(x_train, s_train, "+")
        # plt.show()

        kern = GPy.kern.Linear(dim)
        m_dist = GPy.models.GPDistillation(x_train, y_train, s_train, kernel=kern)
        m_dist['mixed_noise.Gaussian_noise.variance'] = 1e-3
        m_dist['mixed_noise.Gaussian_noise.variance'].constrain_fixed()
        m_dist.optimize(messages=verbose)
        y_pred, _ = m_dist.predict(x_test)
        y_pred = np.where(y_pred > 0.5, 1, -1)
        dist_result.append(accuracy_score(y_test, y_pred))
        if verbose:
            print(m_dist)
            print("distillation", dist_result[-1], "\n")

        m_plain = GPy.models.GPClassification(x_train, y_train, kernel=kern)
        m_plain.optimize(messages=verbose)
        y_pred, _ = m_plain.predict(x_test)
        y_pred = np.where(y_pred > 0.5, 1, -1)
        plain_result.append(accuracy_score(y_test, y_pred))
        if verbose:
            print(m_plain)
            print('plain', plain_result[-1], "\n")

        kern = GPy.kern.Linear(1)
        m_priv = GPy.models.GPClassification(s_train, y_train, kernel=kern)
        m_priv.optimize(messages=verbose)
        y_pred, _ = m_priv.predict(s_test)
        y_pred = np.where(y_pred > 0.5, 1, -1)
        priv_result.append(accuracy_score(y_test, y_pred))
        if verbose:
            print(m_priv)
            print("privileged", priv_result[-1], "\n")

    print("distillation", np.mean(dist_result), np.std(dist_result))
    print("plain", np.mean(plain_result), np.std(plain_result))
    print("privileged", np.mean(priv_result), np.std(priv_result))


if __name__ == "__main__":
    toy1(rep=2)