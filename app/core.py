"""
initialize
while metric_not_met:
    generate new samples
    evaluate new samples
    retrain surrogate

assumption - one problem is run at a time
"""
# Import custom packages
from model_class import Model
from settings import load_settings,settings,check_valid_settings

problem_id = 13

# Initialize the settings
settings.update(load_settings("app",problem_id))
check_valid_settings()

# Initialize the model
model = Model()

# Surrogate training loop
while not model.optimization_converged:

    if settings["surrogate"]["surrogate"]:
        while not model.trained:
            if not model.retraining:
                model.sample()
                model.evaluate()
            model.load_results()
            model.train()
            model.surrogate_convergence()

    ##        import matplotlib.pyplot as plt
    ##        plt.scatter(model.data.input[:,0],model.data.input[:,1])
    ##        plt.show()

    if settings["optimization"]["optimize"]:
        # Solve the optimiaztion problem
        model.optimize()

        # Verify whether the optimization result agrees with original model
        model.verify()
    else:
        break

input("Ended")

