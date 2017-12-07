import warnings
import multiprocessing
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from ..core import Model
from .. import kern
from ..likelihoods import Bernoulli
from ..likelihoods.link_functions import Heaviside
from ..inference.latent_function_inference.posterior import PosteriorEP

epsilon = np.finfo(np.float64).eps


class NegativeVarianceWarning(Warning):
    ...


class GPPrivPlus(Model):
    def __init__(self, X, Y, Xstar, kernel=None, kernel_star=None,
                 mean=None, mean_star=None, max_iter=None, parallel_update=True, ignore_warnings=True):
        super(GPPrivPlus, self).__init__("gp_priv_plus")

        self.X = X
        self.Y = np.where(
            Y > 0, np.ones_like(Y, dtype=int), -np.ones_like(Y, dtype=int))
        self.Xstar = Xstar
        if kernel is None:
            self.kernel = kern.RBF(X.shape[1])
        else:
            self.kernel = kernel

        if kernel_star is None:
            self.kernel_star = kern.RBF(Xstar.shape[1])
        else:
            self.kernel_star = kernel_star

        self.mean = mean
        self.mean_star = mean_star

        self.link_parameter(self.kernel)
        self.link_parameter(self.kernel_star)

        if mean is not None:
            self.link_parameter(self.mean)

        if mean_star is not None:
            self.link_parameter(self.mean_star)

        self.site = self.site_star = None
        self.posterior = None
        self._log_marginal_likelihood = None
        self.grad_dict = None

        self.max_iter = 100 if max_iter is None else max_iter
        self.parallel_update = parallel_update
        self.damping = 0.9

        if ignore_warnings:
            warnings.simplefilter('ignore', NegativeVarianceWarning)

        self.pool = multiprocessing.Pool() if parallel_update else None

    def parameters_changed(self):
        K = self.kernel.K(self.X)
        Kstar = self.kernel_star.K(self.Xstar)
        if self.mean is None:
            mean = np.zeros(self.X.shape[0])
        else:
            mean = self.mean.f(self.X).flatten()

        if self.mean_star is None:
            mean_star = np.zeros(self.Xstar.shape[0])
        else:
            mean_star = self.mean_star.f(self.Xstar).flatten()

        self.posterior, self._log_marginal_likelihood, self.grad_dict = self._ep(
            K, Kstar, self.Y, mean, mean_star)
        self.kernel.update_gradients_full(self.grad_dict['dL_dK'], self.X)
        self.kernel_star.update_gradients_full(self.grad_dict['dL_dKstar'], self.Xstar)

        if self.mean is not None:
            self.mean.update_gradients(self.grad_dict['dL_dm'], self.X)

        if self.mean_star is not None:
            self.mean_star.update_gradients(self.grad_dict['dL_dm_star'], self.Xstar)

    def log_likelihood(self):
        return self._log_marginal_likelihood

    def predict(self, Xnew, full_cov=False, include_likelihood=True):
        mu, var = self.posterior._raw_predict(self.kernel, Xnew, pred_var=self.X, full_cov=full_cov)
        if include_likelihood:
            mu, var = Bernoulli(Heaviside()).predictive_values(mu, var, full_cov=full_cov)
        return mu, var

    def _ep(self, K, Kstar, Y, mean, mean_star):
        n_data = Y.shape[0]
        site, site_star, post, post_star = self._init_ep(K, Kstar, mean, mean_star)
        log_Z = np.zeros(n_data)
        converged = False

        for loop in range(self.max_iter):
            print(loop, "th iteration")
            old_site = site.copy()
            old_site_star = site_star.copy()

            if not self.parallel_update:
                order = np.random.permutation(n_data)
                for i in order:
                    cav = CavParam(i, site, post)
                    cav_star = CavParam(i, site_star, post_star)

                    nu, tau, nu_star, tau_star, log_Z[i] = _next_site(cav, cav_star, Y.flat[i])

                    site.update(i, nu, tau, damping=self.damping)
                    site_star.update(i, nu_star, tau_star, damping=self.damping)

                    post.local_update(i, old_site, site)
                    post_star.local_update(i, old_site_star, site_star)
            else:
                args = [(CavParam(i, site, post), CavParam(i, site_star, post_star), Y[i])
                        for i in range(n_data)]
                nu, tau, nu_star, tau_star, log_Z = zip(*self.pool.starmap(_next_site, args))
                site.update_parallel(np.array(nu), np.array(tau), damping=self.damping)
                site_star.update_parallel(np.array(nu_star), np.array(tau_star), damping=self.damping)

            post.full_update(K, mean, site)
            post_star.full_update(Kstar, mean_star, site_star)

            if self._converged(site, site_star, old_site, old_site_star):
                converged = True
                break

        if not converged:
            warnings.warn("Iteration count reached maximum %d" % self.max_iter)

        self.site = site
        self.site_star = site_star

        sqrt_tau = np.sqrt(site.tau)
        B = np.eye(site.n) + sqrt_tau[:, None] * K * sqrt_tau
        Lsqrt_tau = np.linalg.solve(np.linalg.cholesky(B), np.diag(sqrt_tau))
        Winv = Lsqrt_tau.T.dot(Lsqrt_tau)
        d = site.nu - site.tau * mean
        alpha = (d - sqrt_tau * np.linalg.solve(B, sqrt_tau * K.dot(d))).reshape(-1, 1)
        posterior = PosteriorEP(woodbury_inv=Winv, woodbury_vector=alpha, K=K)
        # posterior = PosteriorEP(mean=post.mean, cov=post.cov, K=K)
        log_z0 = self._log_marginal_likelihood_without_constant(post, site, K, mean)
        log_z1 = self._log_marginal_likelihood_without_constant(post_star, site_star, Kstar, mean_star)
        log_marginal_likelihood = float(
            np.sum(log_Z) + log_z0 + log_z1
        )

        grad_dict = {'dL_dK': site.dlml_dK(K, mean),
                     'dL_dKstar': site_star.dlml_dK(Kstar, mean_star),
                     'dL_dm': site.dlml_dm(K, mean),
                     'dL_dm_star': site.dlml_dm(Kstar, mean_star)}
        return posterior, log_marginal_likelihood, grad_dict

    def _init_ep(self, K, Kstar, mean, mean_star, force_init=False, damping=0.9):
        n = K.shape[0]
        if force_init or self.site is None:
            return SiteParam(n), SiteParam(n), PostParam(K, mean), PostParam(Kstar, mean_star)
        else:
            site = SiteParam(n).init_damping(self.site, damping)
            site_star = SiteParam(n).init_damping(self.site_star, damping)
            return (site, site_star,
                    PostParam(K, mean, site),
                    PostParam(Kstar, mean_star, site_star))

    def _converged(self, site, site_star, old_site, old_site_star):
        if old_site is None or old_site_star is None:
            return False

        return (site.is_close(old_site) and
                site_star.is_close(old_site_star))

    def _log_marginal_likelihood_without_constant(self, post, site, K, mean):
        cav_var = 1 / (1 / np.diag(post.cov) - site.tau)
        cav_mean = cav_var * (post.mean / np.diag(post.cov) - site.nu)
        sqrt_tau = np.sqrt(site.tau)
        chol_B = np.linalg.cholesky(np.eye(site.n) + sqrt_tau[:, None] * K * sqrt_tau)
        alpha = np.linalg.solve(chol_B, sqrt_tau * K.dot(site.nu))
        beta = np.linalg.solve(chol_B, sqrt_tau * mean)
        return (0.5 * np.sum(np.log(1+site.tau*cav_var))
                - np.sum(np.log(np.diag(chol_B)))
                + 0.5 * site.nu.dot(K.dot(site.nu))
                - 0.5 * alpha.dot(alpha)
                - 0.5 * np.sum(site.nu**2/(1/cav_var+site.tau))
                + 0.5 * np.sum(cav_mean**2/(1/cav_var+site.tau)*site.tau/cav_var)
                - np.sum(cav_mean/(1/cav_var+site.tau)/cav_var*site.nu)
                + mean.dot(site.nu)
                - alpha.dot(beta)
                - 0.5 * beta.dot(beta)
                - 0.5 * site.n * np.log(2 * np.pi))

    def to_dict(self):
        return self._to_dict()

    def save_model(self, output_filename, compress=True, save_data=True):
        ...

def _next_site(cav, cav_star, y):
    Z = quad(
        lambda g: (norm.cdf(y * cav.mean / np.sqrt(cav.var + np.exp(g))) *
                   norm.pdf(g, cav_star.mean, np.sqrt(cav_star.var))),
        -np.inf, np.inf)[0]
    dZdm = quad(
        lambda g: (y / np.sqrt(cav.var + np.exp(g)) *
                   norm.pdf(y * cav.mean / np.sqrt(cav.var + np.exp(g))) *
                   norm.pdf(g, cav_star.mean, np.sqrt(cav_star.var))),
        -np.inf, np.inf)[0]
    d2Zdm2 = quad(
        lambda g: (-y * cav.mean / np.sqrt(cav.var + np.exp(g)) ** 3 *
                   norm.pdf(y * cav.mean / np.sqrt(cav.var + np.exp(g))) *
                   norm.pdf(g, cav_star.mean, np.sqrt(cav_star.var))),
        -np.inf, np.inf)[0]
    dZdm_star = quad(
        lambda g: (norm.cdf(y * cav.mean / np.sqrt(cav.var + np.exp(g))) *
                   (g - cav_star.mean) / cav_star.var *
                   norm.pdf(g, cav_star.mean, np.sqrt(cav_star.var))),
        -np.inf, np.inf)[0]
    d2Zdm_star2 = quad(
        lambda g: (norm.cdf(y * cav.mean / np.sqrt(cav.var + np.exp(g))) *
                   ((g - cav_star.mean) ** 2 / cav_star.var - 1) / cav_star.var *
                   norm.pdf(g, cav_star.mean, np.sqrt(cav_star.var))),
        -np.inf, np.inf)[0]
    nu, tau = _next_site_common(Z, dZdm, d2Zdm2, cav)
    nu_star, tau_star = _next_site_common(Z, dZdm_star, d2Zdm_star2, cav_star)
    if tau < epsilon:
        warnings.warn("Tau of site distribution %f < 0" % tau,
                      NegativeVarianceWarning)
        tau = epsilon

    if tau_star < epsilon:
        warnings.warn("Tau of noise site distribution %f < 0" % tau_star,
                      NegativeVarianceWarning)
        tau_star = epsilon

    return nu, tau, nu_star, tau_star, np.log(Z)


def _next_site_common(Z, dZdm, d2Zdm2, cav):
    alpha = dZdm / Z
    beta = d2Zdm2 / Z - alpha**2
    nu = (alpha - cav.mean * beta) / (1 + cav.var * beta)
    tau = - beta / (1 + cav.var * beta)
    # z = (np.exp(-0.5 * cav.mean * alpha**2 / (2 + cav.mean * beta)) /
    #      np.sqrt(2*np.pi*cav.var*(2+cav.var*beta)))
    return nu, tau


class SiteParam:
    def __init__(self, n, epsilon=1e-6, rtol=1e-5, atol=1e-8):
        self.n = n
        self.nu = np.zeros(n)
        self.tau = np.zeros(n)
        self.rtol = rtol
        self.atol = atol
        self.epsilon = epsilon

    def init_damping(self, other, damping):
        self.n = other.n
        self.nu = damping * other.nu
        self.tau = damping * other.tau
        return self

    def copy(self):
        site = SiteParam(self.n)
        site.nu = self.nu.copy()
        site.tau = self.tau.copy()
        return site

    def is_close(self, other):
        #return (np.allclose(self.nu, other.nu, rtol=self.rtol, atol=self.atol) and
        #        np.allclose(self.tau, other.tau, rtol=self.rtol, atol=self.atol))
        return (np.mean((self.nu-other.nu)**2) < self.epsilon and
                np.mean((self.tau-other.tau)**2) < self.epsilon)

    def update(self, i, nu, tau, damping=1.0):
        self.nu[i] = damping * nu + (1 - damping) * self.nu[i]
        self.tau[i] = damping * tau + (1 - damping) * self.tau[i]

    def update_parallel(self, nu, tau, damping=1.0):
        self.nu = damping * nu + (1 - damping) * self.nu
        self.tau = damping * tau + (1 - damping) * self.tau

    def dlml_dK(self, K, mean):
        sqrt_tau = np.sqrt(self.tau)
        B = np.eye(self.n) + sqrt_tau[:, None] * K * sqrt_tau
        d = self.nu - self.tau * mean
        b = d - sqrt_tau * np.linalg.solve(B, sqrt_tau * K.dot(d))
        V = np.linalg.solve(np.linalg.cholesky(B), np.diag(sqrt_tau))
        return 0.5 * (b[:, None].dot(b[None, :]) - V.T.dot(V))

    def dlml_dm(self, K, mean):
        sqrt_tau = np.sqrt(self.tau)
        B = np.eye(self.n) + sqrt_tau[:, None] * K * sqrt_tau
        d = self.nu - self.tau * mean
        return (d - sqrt_tau * np.linalg.solve(B, sqrt_tau * K.dot(d)))[:, None]


class PostParam:
    def __init__(self, K, mean, site=None):
        if site is None:
            self.mean = mean.copy()
            self.cov = K.copy()
        else:
            self.full_update(K, mean, site)

    def local_update(self, i, site, old_site):
        tau_delta = site.tau - old_site.tau
        nu_delta = site.nu - old_site.nu
        si = tau_delta / (1 + tau_delta * self.cov[i, i])
        cov_diff = - si * self.cov[i, :, None].dot(self.cov[None, i])
        mean_diff = -(si * (self.mean[i]+self.cov[i, i]*nu_delta) - nu_delta) * self.cov[i]
        self.cov += cov_diff
        self.mean += mean_diff
        return self

    def full_update(self, K, mean, site):
        sqrt_tau = np.sqrt(site.tau)
        B = np.eye(site.n) + sqrt_tau[:, None] * K * sqrt_tau
        V = np.linalg.solve(np.linalg.cholesky(B), sqrt_tau[:, None] * K)
        self.cov = K - V.T.dot(V)
        self.mean = self.cov.dot(site.nu) + mean - K.dot(sqrt_tau * np.linalg.solve(B, sqrt_tau * mean))
        return self


class CavParam:
    def __init__(self, i, site, post):
        self.var = 1/(1/post.cov[i][i] - site.tau[i])
        self.mean = self.var * (post.mean[i]/post.cov[i][i] - site.nu[i])
