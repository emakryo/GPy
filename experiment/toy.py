import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import GPy
import datasets


def toy1(n_train=200, n_test=10000, dim=50, rep=100, verbose=False):
    print("toy1")
    dist_result = []
    plain_result = []
    priv_result = []
    for _ in range(rep):
        x, y, s = datasets.toy1(n_train+n_test, dim=dim)
        x_train, x_test, y_train, y_test, s_train, s_test = train_test_split(x, y, s, test_size=n_test)

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


def toy2(n_train=200, n_test=10000, dim=50, rep=100, verbose=False):
    print("toy2")
    dist_result = []
    plain_result = []
    priv_result = []

    for _ in range(rep):
        x, y, s = datasets.toy2(n_train+n_test, dim)
        x_train, x_test, y_train, y_test, s_train, s_test = train_test_split(x, y, s, test_size=n_test)

        m_plain = GPy.models.GPClassification(x_train, y_train, kernel=GPy.kern.Linear(dim), name="plain")
        m_plain.optimize(messages=verbose)
        y_pred, _ = m_plain.predict(x_test)
        y_pred = np.where(y_pred > 0.5, 1, -1)
        plain_result.append(accuracy_score(y_test, y_pred))
        if verbose:
            print(m_plain)
            print('plain', plain_result[-1], "\n")

        m_priv = GPy.models.GPClassification(s_train, y_train, kernel=GPy.kern.Linear(dim), name="privileged")
        m_priv.optimize(messages=verbose)
        y_pred, _ = m_priv.predict(s_test)
        y_pred = np.where(y_pred > 0.5, 1, -1)
        f_train, v_train = m_priv.predict_noiseless(s_train)
        priv_result.append(accuracy_score(y_test, y_pred))
        if verbose:
            print(m_priv)
            print("privileged", priv_result[-1], "\n")

        m_dist = GPy.models.GPDistillation(x_train, y_train, f_train, kernel=GPy.kern.Linear(dim))
        m_dist.optimize(messages=verbose)
        y_pred, _ = m_dist.predict(x_test)
        y_pred = np.where(y_pred > 0.5, 1, -1)
        dist_result.append(accuracy_score(y_test, y_pred))
        if verbose:
            print(m_dist)
            print("distillation", dist_result[-1], "\n")

    print("distillation", np.mean(dist_result), np.std(dist_result))
    print("plain", np.mean(plain_result), np.std(plain_result))
    print("privileged", np.mean(priv_result), np.std(priv_result))


def toy3(n_train=200, n_test=10000, dim=50, rep=100, verbose=False):
    print('toy3')
    dist_result = []
    plain_result = []
    priv_result = []

    for _ in range(rep):
        x, y, s = datasets.toy3(n_train+n_test, dim)
        x_train, x_test, y_train, y_test, s_train, s_test = train_test_split(x, y, s, test_size=n_test)

        m_plain = GPy.models.GPClassification(x_train, y_train, kernel=GPy.kern.Linear(dim), name="plain")
        m_plain.optimize(messages=verbose)
        y_pred, _ = m_plain.predict(x_test)
        y_pred = np.where(y_pred > 0.5, 1, -1)
        plain_result.append(accuracy_score(y_test, y_pred))
        if verbose:
            print(m_plain)
            print('plain', plain_result[-1], "\n")


        m_priv = GPy.models.GPClassification(s_train, y_train, kernel=GPy.kern.Linear(s_train.shape[1]),
                                             name="privileged")
        m_priv.optimize(messages=verbose)
        y_pred, _ = m_priv.predict(s_test)
        y_pred = np.where(y_pred > 0.5, 1, -1)
        f_train, v_train = m_priv.predict_noiseless(s_train)
        priv_result.append(accuracy_score(y_test, y_pred))
        if verbose:
            print(m_priv)
            print("privileged", priv_result[-1], "\n")

        m_dist = GPy.models.GPDistillation(x_train, y_train, f_train, kernel=GPy.kern.Linear(dim))
        m_dist['mixed_noise.Gaussian_noise.variance'] = 1e-3
        m_dist['mixed_noise.Gaussian_noise.variance'].constrain_fixed()
        m_dist.optimize(messages=verbose)
        y_pred, _ = m_dist.predict(x_test)
        y_pred = np.where(y_pred > 0.5, 1, -1)
        dist_result.append(accuracy_score(y_test, y_pred))
        if verbose:
            print(m_dist)
            print("distillation", dist_result[-1], "\n")

    print("distillation", np.mean(dist_result), np.std(dist_result))
    print("plain", np.mean(plain_result), np.std(plain_result))
    print("privileged", np.mean(priv_result), np.std(priv_result))


def toy4(n_train=200, n_test=10000, dim=50, rep=100, verbose=False):
    print('toy4')
    dist_result = []
    plain_result = []
    priv_result = []

    for _ in range(rep):
        x, y, s = datasets.toy4(n_train+n_test, dim)
        x_train, x_test, y_train, y_test, s_train, s_test = train_test_split(x, y, s, test_size=n_test)

        m_plain = GPy.models.GPClassification(x_train, y_train, kernel=GPy.kern.Linear(dim), name="plain")
        m_plain.optimize(messages=verbose)
        y_pred, _ = m_plain.predict(x_test)
        y_pred = np.where(y_pred > 0.5, 1, -1)
        plain_result.append(accuracy_score(y_test, y_pred))
        if verbose:
            print(m_plain)
            print('plain', plain_result[-1], "\n")


        m_priv = GPy.models.GPClassification(s_train, y_train, kernel=GPy.kern.Linear(s_train.shape[1]),
                                             name="privileged")
        m_priv.optimize(messages=verbose)
        y_pred, _ = m_priv.predict(s_test)
        y_pred = np.where(y_pred > 0.5, 1, -1)
        f_train, v_train = m_priv.predict_noiseless(s_train)
        priv_result.append(accuracy_score(y_test, y_pred))
        if verbose:
            print(m_priv)
            print("privileged", priv_result[-1], "\n")

        m_dist = GPy.models.GPDistillation(x_train, y_train, f_train, kernel=GPy.kern.Linear(dim))
        m_dist.optimize(messages=verbose)
        y_pred, _ = m_dist.predict(x_test)
        y_pred = np.where(y_pred > 0.5, 1, -1)
        dist_result.append(accuracy_score(y_test, y_pred))
        if verbose:
            print(m_dist)
            print("distillation", dist_result[-1], "\n")

    print("distillation", np.mean(dist_result), np.std(dist_result))
    print("plain", np.mean(plain_result), np.std(plain_result))
    print("privileged", np.mean(priv_result), np.std(priv_result))

if __name__ == "__main__":
    toy4(rep=2, verbose=True)
