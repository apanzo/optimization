"""
initialize
while metric_not_met:
    generate new samples
    evaluate new samples
    retrain surrogate
    check metric

assumption - one problem is run at a time
"""
# Import custom packages
from model_class import Model
from settings import load_settings,settings,check_valid_settings

# Initialize the settings
settings.update(load_settings("app","06-tnk"))
check_valid_settings()

# Initialize the model
model = Model()

# Surrogate training loop
while not model.optimization_converged:

    while not model.trained:
        model.sample()
        model.evaluate()
        model.load_results()
        model.train()
        model.sampling_iterations += 1
        model.surrogate_convergence()

    if settings["optimization"]["optimize"]:
        # Solve the optimiaztion problem
        model.optimize()

        # Verify whether the optimization result agrees with original model
        model.verify()
    else:
        break

if settings["optimization"]["optimize"]:
    print(model.optimization_error)
    import numpy as np
    print(np.min(model.optimization_error))
    print(np.max(model.optimization_error))
    print(np.mean(model.optimization_error,0))


input("Ended")
