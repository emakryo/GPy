# Copyright (c) 2012-2014 The GPy authors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from scipy import stats, special
from . import link_functions
from .likelihood import Likelihood
from .gaussian import Gaussian
from ..core.parameterization import Param
from paramz.transformations import Logexp
from ..core.parameterization import Parameterized
import itertools


class MixedNoise(Likelihood):
    def __init__(self, likelihoods_list, name='mixed_noise', noise_index=None):
        #NOTE at the moment this likelihood only works for using a list of gaussians
        super(Likelihood, self).__init__(name=name)

        self.link_parameters(*likelihoods_list)
        self.likelihoods_list = likelihoods_list
        self.log_concave = False
        self.not_block_really = False

    def gaussian_variance(self, Y_metadata):
        ind = Y_metadata['output_index'].flatten()
        assert all([isinstance(self.likelihoods_list[i], Gaussian) for i in np.unique(ind)])
        variance = np.zeros(ind.size)
        for j, lik in enumerate(self.likelihoods_list):
            if j not in np.unique(ind):
                continue
            variance[ind==j] = lik.variance
        return variance

    def betaY(self,Y,Y_metadata):
        #TODO not here.
        return Y/self.gaussian_variance(Y_metadata=Y_metadata)[:,None]

    def update_gradients(self, gradients):
        self.gradient = gradients

    def exact_inference_gradients(self, dL_dKdiag, Y_metadata):
        idx = Y_metadata['output_index'].flatten()
        ret = np.zeros(self.size)
        if self.size == 0:
            return ret

        param_idx = 0
        uniq_idx = np.unique(idx)
        for i, lik in enumerate(self.likelihoods_list):
            if lik.size == 0:
                continue

            if i not in uniq_idx:
                param_idx += lik.size
                continue

            ret[param_idx:param_idx+lik.size] = lik.exact_inference_gradients(
                dL_dKdiag[idx==i], {k: v[idx==i] for k, v in Y_metadata.items()}
            )
            param_idx += lik.size

        return ret

    def predictive_values(self, mu, var, full_cov=False, Y_metadata=None):
        if all([isinstance(l, Gaussian) for l in self.likelihoods_list]):
            ind = Y_metadata['output_index'].flatten()
            _variance = np.array([self.likelihoods_list[j].variance for j in ind ])
            if full_cov:
                var += np.eye(var.shape[0])*_variance
            else:
                var += _variance
            return mu, var
        else:
            assert full_cov is False
            return super(MixedNoise, self).predictive_values(mu, var, full_cov=False, Y_metadata=Y_metadata)

    def predictive_mean(self, mu, variance, Y_metadata=None):
        return self._each_likelihood('predictive_mean', [mu, variance], Y_metadata=Y_metadata)

    def predictive_variance(self, mu, variance, predictive_mean=None, Y_metadata=None):
        return self._each_likelihood('predictive_variance', [mu, variance, predictive_mean], Y_metadata=Y_metadata)

    def predictive_quantiles(self, mu, var, quantiles, Y_metadata):
        assert all([isinstance(l, Gaussian) for l in self.likelihoods_list])
        ind = Y_metadata['output_index'].flatten()
        outputs = np.unique(ind)
        Q = np.zeros( (mu.size,len(quantiles)) )
        for j in outputs:
            q = self.likelihoods_list[j].predictive_quantiles(mu[ind==j,:],
                var[ind==j,:],quantiles,Y_metadata=None)
            Q[ind==j,:] = np.hstack(q)
        return [q[:,None] for q in Q.T]

    def samples(self, gp, Y_metadata):
        """
        Returns a set of samples of observations based on a given value of the latent variable.

        :param gp: latent variable
        """
        N1, N2 = gp.shape
        Ysim = np.zeros((N1,N2))
        ind = Y_metadata['output_index'].flatten()
        for j in np.unique(ind):
            flt = ind==j
            gp_filtered = gp[flt,:]
            n1 = gp_filtered.shape[0]
            lik = self.likelihoods_list[j]
            _ysim = np.array([np.random.normal(lik.gp_link.transf(gpj), scale=np.sqrt(lik.variance), size=1) for gpj in gp_filtered.flatten()])
            Ysim[flt,:] = _ysim.reshape(n1,N2)
        return Ysim

    def moments_match_ep(self, Y_i, tau_i, v_i, Y_metadata_i=None):
        if 'output_index' in Y_metadata_i:
            output_index = Y_metadata_i['output_index'][0]
        else:
            raise ValueError("Index is not specified")

        return self.likelihoods_list[output_index].moments_match_ep(Y_i, tau_i, v_i)

    def _each_likelihood(self, func_name, args, Y_metadata):
        output_index = Y_metadata['output_index'].flatten()
        if 'debug' in Y_metadata:
            print(func_name, args[0].shape, args[1].shape)

        if np.isscalar(args[0]) or np.isscalar(args[1]):
            return self.likelihoods_list[output_index[0]].__getattribute__(func_name)(*args, Y_metadata=Y_metadata)

        ret = np.zeros_like(args[0])
        for i, lik in enumerate(self.likelihoods_list):
            index = output_index == i
            args_i = [arg[index] if isinstance(arg, np.ndarray) else arg for arg in args]
            Y_metadata_i = {k: v[index] for k, v in Y_metadata.items()}
            ret[index] = lik.__getattribute__(func_name)(*args_i, Y_metadata=Y_metadata_i)

        return ret

    def _each_likelihood_dtheta(self, func_name, args, Y_metadata):
        if 'debug' in Y_metadata:
            print(func_name, args[0].shape, args[1].shape)
        if self.size == 0:
            return np.zeros((0, args[0].shape[0], args[0].shape[1]))
        output_index = Y_metadata['output_index'].flatten()

        if np.isscalar(args[0]) or np.isscalar(args[1]):
            return self.likelihoods_list[output_index[0]].__getattribute__(func_name)(*args, Y_metadata=Y_metadata)

        ret = np.zeros((self.size, args[0].shape[0], args[0].shape[1]))
        if self.size == 0:
            return ret

        param_index = 0
        for i, lik in enumerate(self.likelihoods_list):
            index = output_index == i
            args_i = [arg[index] for arg in args]
            Y_metadata_i = {k: v[index] for k, v in Y_metadata.items()}
            try:
                res = lik.__getattribute__(func_name)(*args_i, Y_metadata=Y_metadata_i)
            except NotImplementedError:
                res = np.zeros((0, np.count_nonzero(index), ret.shape[2]))

            ret[param_index:param_index + res.shape[0], index] = res
            param_index += res.shape[0]

        return ret

    def pdf(self, f, y, Y_metadata=None):
        return self._each_likelihood('pdf', [f, y], Y_metadata=Y_metadata)

    def logpdf(self, f, y, Y_metadata=None):
        return self._each_likelihood('logpdf', [f, y], Y_metadata=Y_metadata)

    def dlogpdf_df(self, f, y, Y_metadata=None):
        return self._each_likelihood('dlogpdf_df', [f, y], Y_metadata=Y_metadata)

    def d2logpdf_df2(self, f, y, Y_metadata=None):
        return self._each_likelihood('d2logpdf_df2', [f, y], Y_metadata=Y_metadata)

    def d3logpdf_df3(self, f, y, Y_metadata=None):
        return self._each_likelihood('d3logpdf_df3', [f, y], Y_metadata=Y_metadata)

    def dlogpdf_dtheta(self, f, y, Y_metadata=None):
        return self._each_likelihood_dtheta('dlogpdf_dtheta', [f, y], Y_metadata=Y_metadata)

    def dlogpdf_df_dtheta(self, f, y, Y_metadata=None):
        return self._each_likelihood_dtheta('dlogpdf_df_dtheta', [f, y], Y_metadata=Y_metadata)

    def d2logpdf_df2_dtheta(self, f, y, Y_metadata=None):
        return self._each_likelihood_dtheta('d2logpdf_df2_dtheta', [f, y], Y_metadata=Y_metadata)

    def logpdf_link(self, inv_link_f, y, Y_metadata=None):
        return self._each_likelihood('logpdf_link', [inv_link_f, y], Y_metadata=Y_metadata)

    def dlogpdf_dlink(self, inv_link_f, y, Y_metadata=None):
        return self._each_likelihood('dlogpdf_dlink', [inv_link_f, y], Y_metadata=Y_metadata)

    def d2logpdf_dlink2(self, inv_link_f, y, Y_metadata=None):
        return self._each_likelihood('d2logpdf_dlink2', [inv_link_f, y], Y_metadata=Y_metadata)

    def d3logpdf_dlink3(self, inv_link_f, y, Y_metadata=None):
        return self._each_likelihood('d3logpdf_dlink3', [inv_link_f, y], Y_metadata=Y_metadata)

    def dlogpdf_link_dtheta(self, inv_link_f, y, Y_metadata=None):
        return self._each_likelihood_dtheta('dlogpdf_link_dtheta', [inv_link_f, y], Y_metadata=Y_metadata)

    def dlogpdf_dlink_dtheta(self, inv_link_f, y, Y_metadata=None):
        return self._each_likelihood_dtheta('dlogpdf_dlink_dtheta', [inv_link_f, y], Y_metadata=Y_metadata)

    def d2logpdf_dlink2_dtheta(self, inv_link_f, y, Y_metadata=None):
        return self._each_likelihood_dtheta('d2logpdf_dlink2_dtheta', [inv_link_f, y], Y_metadata=Y_metadata)

    def ep_gradients(self, Y, cav_tau, cav_v, dL_dKdiag, Y_metadata=None, quad_mode='gk', boost_grad=1.):
        output_index = Y_metadata['output_index'].flatten()
        ret = np.zeros(self.size)
        if self.size == 0:
            return ret

        param_index = 0
        for i, lik in enumerate(self.likelihoods_list):
            index = output_index == i
            Y_metadata_i = {k: v[index] for k, v in Y_metadata.items()}
            ep_grad_i = lik.ep_gradients(Y[index], cav_tau[index], cav_v[index], dL_dKdiag[index],
                                         Y_metadata_i, quad_mode, boost_grad)
            if np.isscalar(ep_grad_i):
                ret[param_index] = ep_grad_i
                param_index += 1
            else:
                ret[param_index:param_index+len(ep_grad_i)] = ep_grad_i
                param_index += len(ep_grad_i)

        return ret
