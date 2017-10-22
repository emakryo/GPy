from ..inference.latent_function_inference import EP
from ..core import GP
from .. import likelihoods
from .. import kern
from .. import util


class GPDistillation(GP):
    """
    Gaussian Process model for distillation from soft labels

    This is a thin wrapper around the models.GP class, with a set of sensible defaults

    :param X: input observations
    :type X: numpy arrays
    :param Y: observed binary labels
    :type Y: numpy arrays
    :param S: observed soft labels
    :tye S: numpy arrays
    :param kernel: a GPy kernel ** DualTask, defaults to RBF ** DualTask
    :type kernel: None | GPy.kernel defaults
    :likelihoods_list: a list of likelihoods, defaults to a pair of Bernoulli & Gaussian likelihoods
    :type likelihoods_list None | a list GPy.likelihoods
    :param name: model name
    :type name: string
    :param kernel_name: name of the kernel
    :type kernel_name: string
    """
    def __init__(self, X, Y, S, kernel=None, likelihoods_list=None,
                 name='GPD', kernel_name='dual_task'):

        # Input and Output
        Xall, Yall, self.output_index = util.multioutput.build_XY([X, X], [Y, S])

        # Kernel
        if kernel is None:
            dim = X.shape[1]
            kernel = kern.RBF(dim)
            kernel = kernel.prod(kern.DualTask(input_dim=dim, active_dims=[dim]), name=kernel_name)

        # Likelihood
        if likelihoods_list is not None:
            assert len(likelihoods_list) == 2, "Invalid likelihoods length %d" % len(likelihoods_list)
        else:
            likelihoods_list = [likelihoods.Bernoulli(), likelihoods.Gaussian()]

        likelihood = util.multioutput.build_likelihood([Y, S], self.output_index, likelihoods_list)

        # Inference
        ep = EP(ep_mode='nested')

        # Miscellaneous
        Y_metadata = {'output_index': self.output_index}

        super(GPDistillation, self).__init__(Xall, Yall, kernel, likelihood, name=name,
                                             Y_metadata=Y_metadata, inference_method=ep)
