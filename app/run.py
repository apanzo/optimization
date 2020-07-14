"""
This is the main module

Notes:
    * It is assumed one problem is run at a time
    * keras tuner logs are not stored in data due to max path length issues in kerastuner
"""
# Import custom packages
from core import Model,Surrogate,Optimization
from settings import settings,update_settings

##import sys
##import warnings
##
##if not sys.warnoptions:
##    warnings.simplefilter("ignore")


def train_surrogate():
    """
    Docstring.
    """    
    # Train surrogate
    while not surrogate.trained:
        surrogate.sample()
        surrogate.evaluate()
        surrogate.load_results()
        surrogate.train()
        surrogate.surrogate_convergence()

def reload_surrogate():
    """
    Docstring
    """
    surrogate.reload()

def optimize(surrogate):
    """
    Docstring.
    """
    # Solve the optimiaztion problem
    optimization.optimize(surrogate)

    # Verify whether the optimization result agrees with original model
    if build_surrogate and not load_surrogate:
        optimization.verify()
        
# Choose problem to solve
problem_id = 10

# Initialize the settings
update_settings(problem_id)

# Initialize the model
model = Model() 

# Check computation setup
build_surrogate = bool(settings["surrogate"]["surrogate"])
load_surrogate = settings["surrogate"]["surrogate"] == "load"
perform_optimization = bool(settings["optimization"]["algorithm"])

# Perform computation
# Surrogate only
if build_surrogate and not perform_optimization:
    surrogate = Surrogate(model)
    train_surrogate()
# Direct optimization
elif perform_optimization and not build_surrogate:
    optimization = Optimization(model)
    optimize(None)
# Surrogate based optimization
elif build_surrogate and perform_optimization:
    # Using trained surrogate
    if load_surrogate:
        surrogate = Surrogate(model)
        reload_surrogate()
        optimization = Optimization(model)
        optimize(surrogate)
    # Make surrogate and then optimize
    else:
        surrogate = Surrogate(model)
        train_surrogate()
        optimization = Optimization(model)
        while not optimization.converged:
            if not surrogate.trained:
                train_surrogate()
            optimize(surrogate)    
else:
    print("There is nothing to perform within this model")

# Save trained surrogate
if build_surrogate and not load_surrogate:
    surrogate.save()

##    surrogate.plot_response(inputs=[1,2],output=1,constants=[1])
##    surrogate.plot_response(inputs=[3],output=1,constants=[1,1])

input("Ended")



