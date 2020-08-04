"""
This module provides surrogate pre-processing.

preprocessing stuff
"""
# Import native packages
import operator
from math import ceil
import os

# Import pypi packages
import numpy as np
from sklearn.metrics import max_error as MAX
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import median_absolute_error as MedAE
from sklearn.metrics import r2_score as R2

# Import custom packages
from core.settings import settings

# Functions
def surrogate_performance(surrogates):
    criterion = settings["data"]["convergence"]
    metrics = np.array([model.metric[criterion] for model in surrogates])

    # Store the metric for sample size determination
    variance = np.var(metrics)
    mean = np.mean(metrics)

    # Output
    path = os.path.join(settings["folder"],"logs","surrogate_CV.txt")

    with open(path, "a") as file:
        np.savetxt(file,metrics,newline=" ",fmt='%.5f')
        file.write("\t")
        file.write("mean")
        file.write("\t")
        file.write("%.5f"%mean)
        file.write("\n")    

    return {"mean":mean,"variance":variance}
    
##
def check_convergence(metrics):
    """
    Need to add convergence if data is loaded and there is no more data to load
    """
    print("###### Evaluating sample size convergence ######")
    trained = False
##    breakpoint()
    direction = convergence_operator()
    threshold = settings["data"]["convergence_limit"]
    
    if settings["data"]["convergence_relative"]:
        window_size = 1 ###########
        if len(metrics) < 2:
            return False
        else:
            diff = np.diff(metrics)

            if direction(np.mean(diff[-window_size:]),0):
                if abs(np.mean(diff[-window_size:])) < threshold:
                    trained = True
    else:
       if direction(metrics[-1],threshold):
                trained = True
##    breakpoint()

    print(f"Sample size convergence metric: {settings['data']['convergence']} - {metrics[-1]}")

    return trained

def verify_results(results,surrogate):
    # Set the optimal solutions as new sample
    results = np.atleast_2d(results)
    no_results = results.shape[0]
    verification_ratio = 0.2
    no_verifications = ceil(no_results*verification_ratio)

    # Randomly select a verification sample
    idx =  np.random.default_rng().choice(no_results,size=(no_verifications),replace=False)
    surrogate.samples = results[idx]
    
    # Evaluate the samples and load the results
    surrogate.evaluate_samples(verify=True) 
    surrogate.load_results(verify=True)

    return idx

def evaluate_metrics(inputs,outputs,predict,requested):
    metrics = {}
    for measure in requested:
        metrics[measure] = defined_metrics[measure](outputs,predict(inputs))

    return metrics

def convergence_operator():
    if settings["data"]["convergence"] in ["mae","mse","medae","max_error"]:
        op = operator.lt
    elif settings["data"]["convergence"] in ["r2","max_iterations"]:
        op = operator.gt
    else:
        raise Exception("Error should have been caught on initialization")

    return op

defined_metrics = {
    "r2": R2,
    "mse": MSE,
    "max_error": MAX,
    "medae": MedAE,
    "mae": MAE}
