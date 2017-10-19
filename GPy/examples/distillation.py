import numpy as np
import matplotlib.pyplot as plt
import GPy

def illustrate_1d(n_samples=50, random_state=1):
    if isinstance(random_state, np.random.RandomState):
        generator = random_state
    else:
        generator = np.random.RandomState(random_state)

    X = generator.rand(n_samples, 1) * 10
    S = generator.multivariate_normal(np.zeros(n_samples),
            GPy.kern.RBF(1).K(X) + 0.05 * np.eye(n_samples)).reshape(-1, 1)
    Y = np.sign(S)

    n_grid = 50
    Xgrid = np.zeros((n_grid, 2))
    Xgrid[:, 0] = np.linspace(0, 10)

    m = GPy.models.GPDistillation(X, Y, S)
    fm, fv = m.predict_noiseless(Xgrid)

    print(m)
    print(m.likelihood)
    plt.plot(X, S, 'o')
    plt.plot([0, 10], [0, 0])
    plt.plot(Xgrid[:, 0], fm)
    plt.xlim((1, 10))
    plt.show()

if __name__ == "__main__":
    illustrate_1d()
