from .kern import Kern
from ...core.parameterization import Param

from .kern import Kern
import numpy as np
from .. import Coregionalize
from ...core.parameterization import Param
from paramz.transformations import Logexp, Logistic
from ...util.config import config # for assesing whether to use cython
try:
    from . import coregionalize_cython
    config.set('cython', 'working', 'True')
except ImportError:
    config.set('cython', 'working', 'False')

class DualTask(Coregionalize):
    """
    Covariance function for intrinsic/linear coregionalization models

    This covariance has the form:
    .. math::
       \mathbf{B} = \mathbf{W}\mathbf{W}^\top + \text{diag}(kappa)

    An intrinsic/linear coregionalization covariance function of the form:
    .. math::

       k_2(x, y)=\mathbf{B} k(x, y)

    it is obtained as the tensor product between a covariance function
    k(x, y) and B.

    :param output_dim: number of outputs to coregionalize
    :type output_dim: int
    :param rank: number of columns of the W matrix (this parameter is ignored if parameter W is not None)
    :type rank: int
    :param W: a low rank matrix that determines the correlations between the different outputs, together with kappa it forms the coregionalization matrix B
    :type W: numpy array of dimensionality (num_outpus, W_columns)
    :param kappa: a vector which allows the outputs to behave independently
    :type kappa: numpy array of dimensionality  (output_dim, )

    .. note: see coregionalization examples in GPy.examples.regression for some usage.
    """
    def __init__(self, input_dim, rho=None, active_dims=None, name='dual_task'):
        super(Coregionalize, self).__init__(input_dim, active_dims, name=name)
        if rho is None:
            rho = 0.5
        else:
            assert 0 < rho < 1

        self.output_dim = 2
        self.rho = Param('rho', rho, Logistic(lower=0., upper=1.))
        self.link_parameters(self.rho)

    def parameters_changed(self):
        self.B = np.array([[1, self.rho],
                           [self.rho, 1]])

    def update_gradients_full(self, dL_dK, X, X2=None):
        index = np.asarray(X, dtype=np.int)
        if X2 is None:
            index2 = index
        else:
            index2 = np.asarray(X2, dtype=np.int)

        #attempt to use cython for a nasty double indexing loop: fall back to numpy
        if config.getboolean('cython', 'working'):
            dL_dK_small = self._gradient_reduce_cython(dL_dK, index, index2)
        else:
            dL_dK_small = self._gradient_reduce_numpy(dL_dK, index, index2)

        drho = dL_dK_small[0,1] + dL_dK_small[1,0]
        self.rho.gradient = drho

    def update_gradients_diag(self, dL_dKdiag, X):
        self.rho.gradient = 0.
