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
from datamod import scale
from settings import settings

def determine_samples(no_samples,dim_in):
    if no_samples == 0:
        no_new_samples = settings["data"]["default_sample_coef"]*dim_in
    else:
        if settings["data"]["resampling"] == "linear":
            no_new_samples = settings["data"]["resampling_param"]*dim_in
        elif settings["data"]["resampling"] == "geometric":
            no_new_samples  = int(settings["data"]["resampling_param"]*no_samples)
        else:
            raise Exception("Error should have been caught on initialization")

    return no_new_samples

def resample_static(points_new,points_now,range_in):
    """
    Determine the coordinates of the new sample.

    Arguments:
        points_new: number of new samples
        points_now: number of current samples
        sampling: sampling strategy
        dim_in: number of input dimensions
        range_in: range of inputs

    Returns:
        coordinates: coordinates of the new samples
    
    """
    dim_in = range_in.shape[0]
    # Sample
    full_sample = sample(settings["data"]["sampling"],points_now+points_new,dim_in) # unit coordinates
    new_sample = full_sample[points_now:,:] # only picked those that are new
    coordinates = scale(new_sample,range_in) # full coordinates

    return coordinates

def resample_adaptive(points_new,surrogates,data):
    """
    STUFF

    not working for multiple dimensions
    """

    multiplier_proposed = 100
    np_proposed_points = data.input.shape[0]*multiplier_proposed
    
    proposed_samples = sample(settings["data"]["sampling"],np_proposed_points,data.dim_in)
    predictions_list = [sur.predict_values(proposed_samples) for sur in surrogates]
    predictions = np.array(predictions_list)

    coordinates = sample_adaptive(data,proposed_samples,predictions,points_new)

    return coordinates


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
        Grid actually doesnt make full grid
    """

    xlimits = np.tile((-1,1),(n_dim,1))
    if name in samplings.keys():
        sampling = samplings[name](xlimits=xlimits)
    else:
        raise NameError('Sampling not defined')

    return sampling(points)

def sample_adaptive(data,samples,predictions,no_points_new,exploration,exploitation):
    """
    Sampling using an adaptive DOE

    Notes:
        * Make a unit test to check that worst_new is 2D
    """
    adaptive_method = settings["data"]["adaptive"]

    exploration, exploitation = adaptive_methods[adaptive_method]

    if exploration == "nnd":
        exploration_metric = np.array([np.linalg.norm(data.input-sample,axis=1).min() for sample in samples])
    else:
        raise Exception("Exploration method not implemented")

    if exploitation == "variance":
        exploitation_metric = np.var(predictions,axis=0)[:,0]
    else:
        raise Exception("Exploitation method not implemented")

    if adaptive_method == "eason":
        overall_metric = exploration_metric/np.max(exploration_metric) + exploitation_metric/np.max(exploitation_metric)
    else:
        raise Exception("Adaptive method not implemented")
    
    reordered_metrics = np.argpartition(overall_metric, -no_points_new,axis=0)
    candidate_indices = reordered_metrics[-no_points_new:]
    candidates = samples[candidate_indices]

    return candidates

def complete_grid(density,n_dim):
    no_points = density**n_dim

    return sample("grid",no_points,n_dim)

def partial_grid(density,tot_dim,dims):
    partial_sample = complete_grid(density,len(dims))

    sample = np.zeros((partial_sample.shape[0],tot_dim))
    sample[:,dims] = partial_sample

    return sample

class Halton(SamplingMethod):
    """
    Halton sampling.

    References:
        https://gist.github.com/tupui/cea0a91cc127ea3890ac0f002f887bae
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        no_dim = self.options["xlimits"].shape[0]
        if no_dim > 8:
            raise Exception("Halton sampling performs poor for no_dim > 8")

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
