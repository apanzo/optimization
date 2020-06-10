"""
This is the sampling module.

This module provides sampling methods
"""
import numpy as np
from smt.sampling_methods import FullFactorial, LHS, Random
from smt.sampling_methods.sampling_method import SamplingMethod

from .external.halton import halton

def sample(name,points,n_dim):
    """
    Sampling on a unit hypercube [0,1] using a selected DOE.

    Arguments:
        name: sampling strategy
        points: number of requested samples
        n_dim: number of input dimensions

    Raises:
        NameError: if the sampling is not defined
    """
    
    xlimits = np.tile((0,1),(n_dim,1))
    if name in samplings.keys():        
        sampling = samplings[name](xlimits=xlimits)
    else:
        raise NameError('Sampling not defined')

    return sampling(points)

def select_points_adaptive(surrogates,setting,data):

    test_sample = sample(setting.sampling,setting.adaptive_sample,data.dim_in)
    test_pred = [sur.predict_values(test_sample) for sur in surrogates]
    test_np = np.array(test_pred)
    test_variances = np.var(test_np,axis=0)
    worst = test_sample[np.argmax(test_variances)]
    worst_new = test_sample[np.argpartition(test_variances, -setting.resampling_param,axis=0)[-setting.resampling_param:]]

    nnd = [np.linalg.norm(data.input-sample,axis=1).min() for sample in test_sample]
##    breakpoint()

    return worst

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
