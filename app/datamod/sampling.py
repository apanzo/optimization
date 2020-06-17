"""
This is the sampling module.

This module provides sampling methods
"""
# Import pypi packages
import numpy as np
from smt.sampling_methods import FullFactorial, LHS, Random
from smt.sampling_methods.sampling_method import SamplingMethod

# Import 3rd party packages
from .external.halton import halton

# Import custom packages
from settings import settings

def sample(name,points,n_dim):
    """
    Sampling on a unit hypercube [-1,1] using a selected DOE.

    Arguments:
        name: sampling strategy
        points: number of requested samples
        n_dim: number of input dimensions

    Raises:
        NameError: if the sampling is not defined

    Notes:
        Assumes that data is on the [-1,1] range
    """
    
    xlimits = np.tile((-1,1),(n_dim,1))
    if name in samplings.keys():        
        sampling = samplings[name](xlimits=xlimits)
    else:
        raise NameError('Sampling not defined')

    return sampling(points)

def sample_adaptive(data,sample,test_sample,exploration,exploitation,test_np):
    """
    Sampling using an adaptive DOE
    
    Notes:
        * Make a unit test to check that worst_new is 2D
    """
    if exploration == "nnd":
        nnd = [np.linalg.norm(data.input-sample,axis=1).min() for sample in test_sample]
    else:
        raise Exception("Exploration method not implemented")

    if exploitation == "variance":
        test_variances = np.var(test_np,axis=0)
##        worst = test_sample[np.argmax(test_variances)]
        reorder = np.argpartition(test_variances, -settings["data"]["resampling_param"],axis=0)
        few_best = reorder[-settings["data"]["resampling_param"]:]
        worst_new = test_sample[few_best]
        worst_new = worst_new.reshape((settings["data"]["resampling_param"],-1))
    else:
        raise Exception("Exploitation method not implemented")
    
    return worst_new

class Halton(SamplingMethod):
    """
    Halton sampling.

    References:
        https://gist.github.com/tupui/cea0a91cc127ea3890ac0f002f887bae
    """
    
    def _initialize(self):
##        self.options.declare("weights", values=[None], types=[list, np.ndarray])
##        self.options.declare("clip", types=bool)
        pass

    def _compute(self, nt):
        """
        Compute the requested number of sampling points.

        Arguments
        ---------
        nt : int
            Number of points requested.

        Returns
        -------
        ndarray[nt, nx]
            The sampling locations in the input space.
        """
        xlimits = self.options["xlimits"]
        nx = xlimits.shape[0]

        x = halton(nx,nt)

        return x

samplings = {
    "halton": Halton,
    "grid": FullFactorial,
    "lhs": LHS,
    "random": Random}
