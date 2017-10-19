# Copyright (c) 2012 - 2014 the GPy Austhors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from ..models import GPCoregionalizedRegression
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
    :param kernel: a GPy kernel ** Coregionalized, defaults to RBF ** Coregionalized
    :type kernel: None | GPy.kernel defaults
    :likelihoods_pair: a list of likelihoods, defaults to a pair of Bernoulli & Gaussian likelihoods
    :type likelihoods_pair: None | a list GPy.likelihoods
    :param name: model name
    :type name: string
    :param W_rank: number tuples of the corregionalization parameters 'W' (see coregionalize kernel documentation)
    :type W_rank: integer
    :param kernel_name: name of the kernel
    :type kernel_name: string
    """
    def __init__(self, X, Y, S, kernel=None, likelihoods_list=None, name='GPD',W_rank=1,kernel_name='coreg'):

        #Input and Output
        Xall, Yall, self.output_index = util.multioutput.build_XY([X, X], [Y, S])

        #Kernel
        if kernel is None:
            kernel = kern.RBF(Xall.shape[1]-1)
            
            kernel = util.multioutput.ICM(input_dim=Xall.shape[1]-1, num_outputs=2, kernel=kernel, W_rank=1,name=kernel_name)

        #Likelihood
        likelihoods_list = [likelihoods.Bernoulli(), likelihoods.Gaussian()]
        likelihood = util.multioutput.build_likelihood([Y,S],self.output_index,likelihoods_list)

        super(GPDistillation, self).__init__(Xall,Yall,kernel,likelihood, Y_metadata={'output_index':self.output_index}, inference_method=EP())
