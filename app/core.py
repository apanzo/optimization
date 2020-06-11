"""
initialize
while metric_not_met:
    generate new samples
    evaluate new samples
    retrain surrogate
    check metric

assumption - one problem is run at a time
"""

import os

from datamod.visual import ss_convergence
from model_class import Model
from settings import settings

# Initialize the model
model = Model()

while not model.optimization_converged:
    # Surrogate training loop
    while not model.trained:
        model.sample()
        model.evaluate()
        model.load_results()
        model.train()
##        breakpoint()
        
        model.sampling_iterations += 1

        if model.sampling_iterations == 2:
            print("Surrogate converged")
            model.trained = True

    # Plot the sample size convergence
    ##ss_convergence(model)
    ##compare()

    if settings["optimization"]["optimize"]:
        # Solve the optimiaztion problem
        model.optimize()

        # Verify whether the optimization result agrees with original model
        model.verify()
    else:
        break
