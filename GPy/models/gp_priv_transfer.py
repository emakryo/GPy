import numpy as np
from . import GPClassification
from ..inference.latent_function_inference import EP, EPConditional
from ..core import GP
from .. import likelihoods
from ..likelihoods.link_functions import Identity
from .. import kern
from .. import util


class GPPrivTransfer(GP):
    """
    Gaussian Process model for transferring privilege information from soft labels

    This is a thin wrapper around the models.GP class, with a set of sensible defaults

    :param X: input observations
    :type X: numpy arrays
    :param Y: observed binary labels
    :type Y: numpy arrays
    :param S: observed soft labels
    :tye S: numpy arrays
    :param kernel: a GPy kernel, defaults to RBF
    :type kernel: None | GPy.kernel defaults
    :likelihoods_list: a list of likelihoods, defaults to a pair of Bernoulli & Gaussian likelihoods
    :type likelihoods_list None | a list GPy.likelihoods
    :param name: model name
    :type name: string
    :param kernel_name: name of the kernel
    :type kernel_name: string
    """
    def __init__(self, X, Y, S, V, posterior=True, kernel=None, gauss_likelihood='post',
                 priv_kernel=None, name='gp_priv_transfer', max_iters=np.inf):

        # If posterior is False, S is assumed to be privileged information and V is ignored
        if not posterior:
            if priv_kernel is None:
                priv_kernel = kern.RBF(S.shape[1])
                priv_kernel.variance.constrain_fixed()

            m_class = GPClassification(S, Y, kernel=priv_kernel)
            m_class.optimize()

            S, V = m_class.predict_noiseless(X)
            self.m_class = m_class
        else:
            self.m_class = None

        # Input and Output
        Xall, Yall, self.output_index = util.multioutput.build_XY([X, X], [Y, S])
        dim = X.shape[1]

        # Kernel
        if kernel is None:
            kernel = kern.RBF(dim)
            kernel.variance.constrain_fixed()

        self.base_kernel = kernel
        kernel = kernel.prod(kern.DualTask(input_dim=1, active_dims=[dim]), name='k')

        # Likelihood
        bernoulli = likelihoods.Bernoulli()
        if gauss_likelihood == 'post':
            gaussian = FixedHeteroscedasticGaussian(V.flatten())
        elif np.isscalar(gauss_likelihood):
            gaussian = likelihoods.Gaussian()
            gaussian.variance.constrain_fixed(gauss_likelihood)
        else:
            gaussian = likelihoods.Gaussian()

        likelihoods_list = [bernoulli, gaussian]

        self.likelihood_list = likelihoods_list
        likelihood = util.multioutput.build_likelihood([Y, S], self.output_index, likelihoods_list)

        # Inference
        ep = EPConditional(ep_mode='nested', max_iters=max_iters)
        # ep = EP(ep_mode='nested', max_iters=max_iters)

        # Miscellaneous
        Y_metadata = {'output_index': self.output_index,
                      'variance': np.r_[np.ones_like(V), V],
                      'conditional_index': self.output_index != 0}

        super(GPPrivTransfer, self).__init__(Xall, Yall, kernel, likelihood, name=name,
                                             Y_metadata=Y_metadata, inference_method=ep)

    def predict(self, Xnew, full_cov=False, Y_metadata=None, kern=None, likelihood=None, include_likelihood=True):
        Xnew, _, output_index = util.multioutput.build_XY([Xnew])
        if Y_metadata is None:
            Y_metadata = {}

        Y_metadata['output_index'] = output_index
        return super(GPPrivTransfer, self).predict(Xnew, full_cov=full_cov, Y_metadata=Y_metadata, kern=kern,
                                                   likelihood=likelihood, include_likelihood=include_likelihood)

    def posterior_samples_f(self, X, size=10, full_cov=True, **predict_kwargs):
        X, _, _ = util.multioutput.build_XY([X])
        return super(GPPrivTransfer, self).posterior_samples_f(X, size=size, full_cov=full_cov, **predict_kwargs)


class FixedHeteroscedasticGaussian(likelihoods.Gaussian):
    def __init__(self, variance):
        gp_link = Identity()
        super(likelihoods.Gaussian, self).__init__(gp_link, 'fhet_Gauss')
        self.variance = variance

    def update_gradients(self, grads):
        ...