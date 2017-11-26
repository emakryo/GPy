import warnings
import numpy as np
from ..core import Model
from .. import kern
from ..inference.latent_function_inference.posterior import PosteriorEP

epsilon = np.finfo(np.float64).eps


class GPPrivPlus(Model):
    def __init__(self, X, Y, Xstar, kernel=None, kernel_star=None, max_iter=None):
        super(GPPrivPlus, self).__init__("gp_priv_plus")

        self.X = X
        self.Y = Y
        self.Xstar = Xstar
        if kernel is None:
            self.kernel = kern.RBF(X.shape[1]) + kern.White(X.shape[1], variance=1e-3)
        else:
            self.kernel = kernel

        if kernel_star is None:
            self.kernel_star = kern.RBF(Xstar.shape[1]) + kern.White(Xstar.shape[1], variance=1e-3)
        else:
            self.kernel_star = kernel_star

        self.link_parameter(self.kernel)
        self.link_parameter(self.kernel_star)

        self.posterior = None
        self._log_marginal_likelihood = None
        self.grad_dict = None

        self.max_iter = 100 if max_iter is None else max_iter

    def parameters_changed(self):
        K = self.kernel.K(self.X)
        Kstar = self.kernel_star.K(self.Xstar)
        self.posterior, self._log_marginal_likelihood, self.grad_dict = self._ep(K, Kstar, self.Y)
        self.kernel.update_gradients_full(self.grad_dict['dL_dK'], self.X)
        self.kernel_star.update_gradients_full(self.grad_dict['dL_dKstar'], self.Xstar)

    def log_likelihood(self):
        return self._log_marginal_likelihood

    def _ep(self, K, Kstar, Y, dumping=0.5):
        site, site_star, post, post_star = self._init_ep(K, Kstar)
        log_Z = np.zeros(site.n)
        converged = False

        for _ in range(self.max_iter):
            old_site = site.copy()
            old_site_star = site_star.copy()

            #order = np.random.permutation(site.n)
            order = range(site.n)
            for i in order:
                cav = CavParam(i, site, post)
                cav_star = CavParam(i, site_star, post_star)

                nu, tau, nu_star, tau_star, log_Z[i] = self._next_site(cav, cav_star, Y.flat[i])

                site.update(i, nu, tau, dumping=dumping)
                site_star.update(i, nu_star, tau_star, dumping=dumping)

                post.local_update(i, old_site, site)
                post_star.local_update(i, old_site_star, site_star)

            post.full_update(K, site)
            post_star.full_update(Kstar, site_star)
            if self._converged(site, site_star, old_site, old_site_star):
                converged = True
                break

        if not converged:
            warnings.warn("Iteration count reached maximum %d"%self.max_iter)

        self.site = site
        self.site_star = site_star

        posterior = PosteriorEP(mean=post.mean, cov=post.cov)
        log_z0 = self._log_marginal_likelihood_without_constant(post, site, K)
        log_z1 = self._log_marginal_likelihood_without_constant(post_star, site_star, Kstar)
        log_marginal_likelihood = float(
            np.sum(log_Z) + log_z0 + log_z1
        )

        grad_dict = {'dL_dK': site.dlml_dK(K),
                     'dL_dKstar': site_star.dlml_dK(Kstar)}
        return posterior, log_marginal_likelihood, grad_dict

    def _init_ep(self, K, Kstar):
        n = K.shape[0]
        return SiteParam(n), SiteParam(n), PostParam(K), PostParam(Kstar)

    def _converged(self, site, site_star, old_site, old_site_star):
        if old_site is None or old_site_star is None:
            return False

        return (site.is_close(old_site) and
                site_star.is_close(old_site_star))

    def _next_site(self, cav, cav_star, y):
        from scipy.integrate import quad
        from scipy.stats import norm
        y = 1 if y > 0 else -1

        Z = quad(
            lambda g: (norm.cdf(y * cav.mean / np.sqrt(cav.var + np.exp(g))) *
                       norm.pdf(g, cav_star.mean, np.sqrt(cav_star.var))),
            -np.inf, np.inf)[0]
        dZdm = quad(
            lambda g: (y / np.sqrt(cav.var + np.exp(g)) *
                       norm.pdf(y * cav.mean / np.sqrt(cav.var + np.exp(g))) *
                       norm.pdf(g, cav_star.mean ,np.sqrt(cav_star.var))),
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
        nu, tau = self._next_site_common(Z, dZdm, d2Zdm2, cav)
        nu_star, tau_star = self._next_site_common(Z, dZdm_star, d2Zdm_star2, cav_star)
        if tau < epsilon:
            warnings.warn("Tau of site distribution %f < 0"%tau)
            tau = epsilon

        if tau_star < epsilon:
            warnings.warn("Tau of noise site distribution %f < 0"%tau_star)
            tau_star = epsilon

        return nu, tau, nu_star, tau_star, np.log(Z)

    def _next_site_common(self, Z, dZdm, d2Zdm2, cav):
        alpha = dZdm / Z
        beta = d2Zdm2 / Z - alpha**2
        nu = (alpha - cav.mean * beta) / (1 + cav.var * beta)
        tau = - beta / (1 + cav.var * beta)
        # z = (np.exp(-0.5 * cav.mean * alpha**2 / (2 + cav.mean * beta)) /
        #      np.sqrt(2*np.pi*cav.var*(2+cav.var*beta)))
        return nu, tau

    def _log_marginal_likelihood_without_constant(self, post, site, K):
        cav_var = 1 / (1 / np.diag(post.cov) - site.tau)
        cav_mean = cav_var * (post.mean / np.diag(post.cov) - site.nu)
        sqrt_tau = np.sqrt(site.tau)
        chol_B = np.linalg.cholesky(np.eye(site.n) + sqrt_tau[:,None] * K * sqrt_tau)
        alpha = np.linalg.solve(chol_B, sqrt_tau * K.dot(site.nu))
        return (0.5 * np.sum(np.log(1+site.tau*cav_var))
                - np.sum(np.log(np.diag(chol_B)))
                + 0.5 * site.nu.dot(K.dot(site.nu))
                - 0.5 * alpha.dot(alpha)
                - 0.5 * np.sum(site.nu**2/(1/cav_var+site.tau))
                + 0.5 * np.sum(cav_mean**2/(1/cav_var+site.tau)*site.tau/cav_var)
                - np.sum(cav_mean/(1/cav_var+site.tau)/cav_var*site.nu)
                - 0.5 * site.n * np.log(2 * np.pi))

    def to_dict(self):
        return self._to_dict()

    def save_model(self, output_filename, compress=True, save_data=True):
        ...


class SiteParam:
    def __init__(self, n, rtol=1e-5, atol=1e-8):
        self.n = n
        self.nu = np.zeros(n)
        self.tau = np.zeros(n)
        self.rtol = rtol
        self.atol = atol

    def copy(self):
        site = SiteParam(self.n)
        site.nu = self.nu.copy()
        site.tau = self.tau.copy()
        return site

    def is_close(self, other):
        return (np.allclose(self.nu, other.nu, rtol=self.rtol, atol=self.atol) and
                np.allclose(self.tau, other.tau, rtol=self.rtol, atol=self.atol))

    def update(self, i, nu, tau, dumping=1.0):
        self.nu[i] = dumping * nu + (1-dumping) * self.nu[i]
        self.tau[i] = dumping * tau + (1-dumping) * self.tau[i]

    def dlml_dK(self, K):
        sqrt_tau = np.sqrt(self.tau)
        B = np.eye(self.n) + sqrt_tau[:, None] * K * sqrt_tau
        b = self.nu - sqrt_tau * np.linalg.solve(B, sqrt_tau * K.dot(self.nu))
        V = np.linalg.solve(np.linalg.cholesky(B), np.diag(sqrt_tau))
        return 0.5 * (b[:, None].dot(b[None, :]) - V.T.dot(V))


class PostParam:
    def __init__(self, K):
        self.mean = np.zeros(K.shape[0])
        self.cov = K.copy()

    def local_update(self, i, site, old_site):
        self.cov -= ((site.tau - old_site.tau) /
                     (1 + (site.tau - old_site.tau) * self.cov[i, i]) *
                     self.cov[i, :, None].dot(self.cov[None, i]))
        self.mean = self.cov.dot(site.nu)

    def full_update(self, K, site):
        sqrt_tau = np.sqrt(site.tau)
        chol_B = np.linalg.cholesky(np.eye(site.n) + sqrt_tau[:, None] * K * sqrt_tau)
        V = np.linalg.solve(chol_B, sqrt_tau[:, None] * K)
        self.cov = K - V.T.dot(V)
        self.mean = self.cov.dot(site.nu)


class CavParam:
    def __init__(self, i, site, post):
        self.var = 1/(1/post.cov[i][i] - site.tau[i])
        self.mean = self.var * (post.mean[i]/post.cov[i][i] - site.nu[i])
