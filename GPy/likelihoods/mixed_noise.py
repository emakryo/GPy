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
        assert all([isinstance(l, Gaussian) for l in self.likelihoods_list])
        ind = Y_metadata['output_index'].flatten()
        variance = np.zeros(ind.size)
        for lik, j in zip(self.likelihoods_list, range(len(self.likelihoods_list))):
            variance[ind==j] = lik.variance
        return variance

    def betaY(self,Y,Y_metadata):
        #TODO not here.
        return Y/self.gaussian_variance(Y_metadata=Y_metadata)[:,None]

    def update_gradients(self, gradients):
        self.gradient = gradients

    def exact_inference_gradients(self, dL_dKdiag, Y_metadata):
        assert all([isinstance(l, Gaussian) for l in self.likelihoods_list])
        ind = Y_metadata['output_index'].flatten()
        return np.array([dL_dKdiag[ind==i].sum() for i in range(len(self.likelihoods_list))])

    def predictive_values(self, mu, var, full_cov=False, Y_metadata=None):
        assert all([isinstance(l, Gaussian) for l in self.likelihoods_list])
        ind = Y_metadata['output_index'].flatten()
        _variance = np.array([self.likelihoods_list[j].variance for j in ind ])
        if full_cov:
            var += np.eye(var.shape[0])*_variance
        else:
            var += _variance
        return mu, var

    def predictive_variance(self, mu, sigma, Y_metadata):
        _variance = self.gaussian_variance(Y_metadata)
        return _variance + sigma**2

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

    def logpdf_link(self, inv_link_f, y, Y_metadata=None):
        output_index = Y_metadata['output_index']

        if np.isscalar(y):
            return self.likelihoods_list[output_index[0]].logpdf_link(inv_link_f, y, Y_metadata)

        ret = np.zeros_like(y)
        Y_metadata_list = []
        for i in range(len(self.likelihoods_list)):
            Y_metadata_i = {}
            for k, v in Y_metadata:
                Y_metadata_i[k] = v[output_index==i]

            Y_metadata_list.append(Y_metadata_i)

        for i, lik in enumerate(self.likelihoods_list):
            index = output_index==i
            ret[index] = lik.logpdf_link(inv_link_f[index], y[index], Y_metadata_list[i])

        return ret

    def dlogpdf_dtheta(self, f, y, Y_metadata=None):
        if self.size == 0:
            return np.zeros((0, f,shape[0], f.shape[1]))

        output_index = Y_metadata['output_index']

        if np.isscalar(y):
            return self.likelihoods_list[output_index[0]].logpdf_link(inv_link_f, y, Y_metadata)

        Y_metadata_list = []
        for i in range(len(self.likelihoods_list)):
            Y_metadata_i = {}
            for k, v in Y_metadata:
                Y_metadata_i[k] = v[output_index==i]

            Y_metadata_list.append(Y_metadata_i)

        ret = np.zeros((self.size, f.shape[0], f.shape[1]))
        param_index = 0
        for i, lik in enumerate(self.likelihoods_list):
            index = output_index==i
            res = lik.dlogpdf_dtheta(inv_link_f[index], y[index], Y_metadata_list[i])
            ret[param_index:param_index+res.shape[0], index] = res
            param_index += res.shape[0]

        return ret
