import numpy as np
from ..inference.latent_function_inference import EP
from ..core import GP
from .. import likelihoods
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
    def __init__(self, X, Y, S, kernel=None, likelihoods_list=None,
                 name='gp_priv_transfer', kernel_name='dual_task', max_iters=np.inf):

        # Input and Output
        Xall, Yall, self.output_index = util.multioutput.build_XY([X, X], [Y, S])
        dim = X.shape[1]

        # Kernel
        if kernel is None:
            kernel = kern.RBF(dim)

        kernel = kernel.prod(kern.DualTask(input_dim=1, active_dims=[dim]), name=kernel_name)

        # Likelihood
        if likelihoods_list is not None:
            assert len(likelihoods_list) == 2, "Invalid likelihoods length %d" % len(likelihoods_list)
        else:
            likelihoods_list = [likelihoods.Bernoulli(), likelihoods.GaussianPV()]

        likelihood = util.multioutput.build_likelihood([Y, S], self.output_index, likelihoods_list)

        # Inference
        ep = EP(ep_mode='nested', max_iters=max_iters)

        # Miscellaneous
        Y_metadata = {'output_index': self.output_index}

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
