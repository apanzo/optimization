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
##settings.update(load_settings("app","00-getting_started"))
##settings.update(load_settings("app","01-squared"))
settings.update(load_settings("app","02-matlab_peaks"))
##settings.update(load_settings("app","03-kursawe_unc"))
##settings.update(load_settings("app","04-bnh_unc"))
##settings.update(load_settings("app","05-tnk_unc"))
##settings.update(load_settings("app","06-tnk"))
##settings.update(load_settings("app","07-osy"))
##settings.update(load_settings("app","08-carside"))
##settings.update(load_settings("app","09-dtlz5"))
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
        model.surrogate_convergence()

    if settings["optimization"]["optimize"]:
        # Solve the optimiaztion problem
        model.optimize()

        # Verify whether the optimization result agrees with original model
        model.verify()
    else:
        break

input("Ended")
