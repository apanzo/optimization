"""
AAA.
"""
# Import native packages
from math import ceil

# Import pypi packages
import numpy as np
from pymoo.factory import get_performance_indicator

# Import custom packages
from core.settings import settings

def calculate_hypervolume(data,ref_point):
    hv = get_performance_indicator("hv", ref_point=ref_point)
    hv = hv.calc(data)
    
    return hv

def verify_results(results,surrogate):
    # Set the optimal solutions as new sample
    results = np.atleast_2d(results)
    no_results = results.shape[0]
    verification_ratio = settings["optimization"]["verification_ratio"]
    no_verifications = ceil(no_results*verification_ratio)

    # Randomly select a verification sample
    idx =  np.random.default_rng().choice(no_results,size=(no_verifications),replace=False)
    surrogate.samples = results[idx]
    
    # Evaluate the samples and load the results
    surrogate.evaluate_samples(verify=True) 
    surrogate.load_results(verify=True)

    return idx
