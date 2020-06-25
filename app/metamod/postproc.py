"""
This module provides surrogate pre-processing.

preprocessing stuff
"""
# Import native packages
from math import ceil

# Import pypi packages
import numpy as np
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import r2_score as R2

# Import custom packages
from settings import settings

# Functions
def select_best_surrogate(surrogates):

    metrics = np.array([sur.metric[settings["surrogate"]["selection_metric"]] for sur in surrogates])

    if settings["surrogate"]["selection_metric"] == "mae":
        best_index = metrics.argmin()
    elif settings["surrogate"]["selection_metric"] == "r2":
        best_index = metrics.argmax()

    best_surrogate = surrogates[best_index]
    
    # Store the metric for sample size determination
    best_surrogate.metric["variance"] = np.var(metrics)
    
    return best_surrogate

##
def check_convergence(metric):
    trained = False

    if settings["data"]["convergence"] in ["mae","variance"]:
        if metric <= settings["data"]["convergence_limit"]:
            trained = True
    elif settings["data"]["convergence"] in ["r2","max_iterations"]:
        if metric >= settings["data"]["convergence_limit"]:
            trained = True
    else:
        raise Exception("Error should have been caught on initialization")

    print(f"Sample size convergence metric: {settings['data']['convergence']} - {metric}")

    return trained

def verify_results(results,surrogate):
    # Set the optimal solutions as new sample
    results = np.atleast_2d(results)
    no_results = results.shape[0]
    verification_ratio = 0.2
    no_verifications = ceil(no_results*verification_ratio)
    idx =  np.random.default_rng().choice(no_results,size=(no_verifications),replace=False)
    surrogate.samples = results[idx]
    
    # Evaluate the samples and load the results
    surrogate.evaluate(verify=True) 
    surrogate.load_results(verify=True)

    return idx

def evaluate_metrics(test_in,test_out,evaluate,requested):
    metrics = {}
    for measure in requested:
        metrics[measure] = defined_metrics[measure](test_out,evaluate(test_in))

    return metrics
          
defined_metrics = {
    "r2": R2,
    "mae": MAE}
