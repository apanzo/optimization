"""
This is the main module

Notes:
    * It is assumed one problem is run at a time
    * keras tuner logs are not stored in data due to max path length issues in kerastuner
"""
# Import custom packages
from core import Model, Surrogate, Optimization
from core.settings import settings, update_settings

def train_surrogate():
    """
    Docstring.
    """    
    # Train surrogate
    while not surrogate.trained:
        surrogate.sample()
        surrogate.evaluate_samples()
        surrogate.load_results()
        surrogate.optimize_hyperparameters()
        surrogate.train()
        surrogate.check_convergence()
        surrogate.report()

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
    optimization.set_problem(surrogate)
    optimization.optimize()

    # Verify whether the optimization result agrees with original model
    if build_surrogate and not (load_surrogate or train_from_data):
        optimization.verify()
    else:
        optimization.converged = True

    # Write
    optimization.report()
        
# Choose problem to solve
problem_id = 0

# Initialize the settings
update_settings(problem_id)

# Initialize the model
model = Model() 

# Check computation setup
build_surrogate = bool(settings["surrogate"]["surrogate"])
load_surrogate = settings["surrogate"]["surrogate"] == "load"
train_from_data = settings["data"]["evaluator"] == "data"
perform_optimization = bool(settings["optimization"]["algorithm"])

# Initialize
if build_surrogate:
    surrogate = Surrogate(model)
if perform_optimization:
    optimization = Optimization(model)


# Perform computation
# Surrogate only
if build_surrogate and not perform_optimization:
    train_surrogate()

# Direct optimization
elif perform_optimization and not build_surrogate:
    optimize(None)

# Surrogate based optimization
elif build_surrogate and perform_optimization:

    # Using trained surrogate
    if load_surrogate:
        reload_surrogate()

    # Make surrogate and then optimize
    else:
        train_surrogate()        

    # Optimize
    while not optimization.converged:
        if not surrogate.trained:
            train_surrogate()
        optimize(surrogate)    

# Otherwise
else:
    print("There is nothing to perform within this model")

# Evaluate benchmark
if settings["data"]["evaluator"] == "benchmark":
    if perform_optimization:
        optimization.benchmark()

if build_surrogate:
    surrogate.plot_response(inputs=[1,2],output=1)
    ##    surrogate.plot_response(inputs=[1,2],output=1,constants=[1])
    ##    surrogate.plot_response(inputs=[3],output=1,constants=[1,1])

# Save trained surrogate
if build_surrogate and not load_surrogate:
    surrogate.save()

input("Ended")
