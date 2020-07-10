"""
This module provides surrogate pre-processing.

preprocessing stuff
"""
# Import native packages
import operator
from math import ceil

# Import pypi packages
import numpy as np
from sklearn.metrics import max_error as MAX
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import median_absolute_error as MedAE
from sklearn.metrics import r2_score as R2

# Import custom packages
from settings import settings

# Functions
def select_best_surrogate(surrogates):

    metrics = np.array([sur.metric[settings["surrogate"]["selection_metric"]] for sur in surrogates])

    argbest, _ = maximize_minimize()
    select_best = argbest(metrics)

    best_surrogate = surrogates[select_best]
    
    # Store the metric for sample size determination
    best_surrogate.metric["variance"] = np.var(metrics)

    return best_surrogate

##
def check_convergence(metrics):
    trained = False

    _, direction = maximize_minimize()
    threshold = settings["data"]["convergence_limit"]
    
    if settings["data"]["convergence_relative"]:
        if len(metrics) < 2:
            return False
        else:
            diff = np.diff(metrics)

            if direction(diff[-1],0):
                if abs(diff[-1]) < threshold:
                    trained = True
    else:
       if direction(metrics[-1],threshold):
                trained = True

    print("###### Evaluating sample size convergence ######")
    print(f"Sample size convergence metric: {settings['data']['convergence']} - {metrics[-1]}")

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

def evaluate_metrics(inputs,outputs,predict,requested):
    metrics = {}
    for measure in requested:
        metrics[measure] = defined_metrics[measure](outputs,predict(inputs))

    return metrics

def maximize_minimize():
    if settings["surrogate"]["selection_metric"] in ["mae","mse","medae","max_error"]:
        target = np.argmin
    elif settings["surrogate"]["selection_metric"] in ["r2"]:
        target = np.argmax
    else:
        raise Exception("Error should have been caught on initialization")

    if settings["data"]["convergence"] in ["mae","mse","medae","max_error"]:
        op = operator.lt
    elif settings["data"]["convergence"] in ["r2","max_iterations"]:
        op = operator.gt
    else:
        raise Exception("Error should have been caught on initialization")

    return target, op
          
defined_metrics = {
    "r2": R2,
    "mse": MSE,
    "max_error": MAX,
    "medae": MedAE,
    "mae": MAE}
