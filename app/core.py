"""
initialize
while metric_not_met:
    generate new samples
    evaluate new samples
    retrain surrofate
    check metric
"""

from datamod.visual import ss_convergence
from model_class import Model
from settings import First, Second

# Initialize the model
setting = Second()
model = Model(setting)

while not model.optimization_converged:
    # Surrogate training loop
    while not model.trained:
        model.sample()
        model.evaluate()
        model.load_results()
        model.train()
##        breakpoint()

        
        model.sampling_iterations += 1

        if model.sampling_iterations == 1:
            print("Surrogate converged")
            model.trained = True

    ##breakpoint()

    # Plot the sample size convergence
    ##ss_convergence(model)
    ##compare()

    if model.setting.optimize:
        # Solve the optimiaztion problem
        model.optimize()

        # Verify whether the optimization result agrees with original model
        model.verify()
    else:
        break
