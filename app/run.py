"""
This is the main module

Notes:
    * It is assumed one problem is run at a time
    * Retraining not applied yet
    * Retraining logic missing as well
    * Pass surrogate setting (kriging)
    * CHeck if shuffle is needed in Kfold
"""
# Import custom packages
from core import Model,Surrogate,Optimization
from settings import settings,update_settings

def train_surrogate():
    """
    Docstring.
    """    
    # Train surrogate
    while not surrogate.trained:
        if True:
##        if not surrogate.retraining or not settings["surrogate"]["append_verification"]:
            surrogate.sample()
            surrogate.evaluate()
        surrogate.load_results()
        surrogate.train()
        surrogate.surrogate_convergence()

def optimize():
    """
    Docstring.
    """
    # Solve the optimiaztion problem
    optimization.optimize()

    # Verify whether the optimization result agrees with original model
    if build_surrogate:
        optimization.verify()
        
# Choose problem to solve
problem_id = 3

# Initialize the settings
update_settings(problem_id)

# Initialize the model
model = Model() 

# Check computation setup
build_surrogate = bool(settings["surrogate"]["surrogate"])
perform_optimization = bool(settings["optimization"]["algorithm"])

# Perform comupation
if build_surrogate and not perform_optimization:
    surrogate = Surrogate(model)
    train_surrogate()
elif perform_optimization and not build_surrogate:
    optimization = Optimization(model,None,None,None)
    optimize()
elif build_surrogate and perform_optimization:
    surrogate = Surrogate(model)
    train_surrogate()
    optimization = Optimization(model,surrogate)
    while not optimization.converged:
        if not surrogate.trained:
            train_surrogate()
        optimize()    
else:
    print("There is nothing to perform within this model")

input("Ended")
