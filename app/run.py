"""
This is the main module

Notes:
    * It is assumed one problem is run at a time
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
        if not surrogate.retraining or not settings["surrogate"]["append_verification"]:
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
    optimization = Optimization(model,model.system.evaluate,None,None,direct=True)
    optimize()
elif build_surrogate and perform_optimization:
    surrogate = Surrogate(model)
    train_surrogate()
    optimization = Optimization(model,surrogate.surrogate.predict_values,surrogate.data.range_out,surrogate)
    while not model.converged:
        if surrogate.retraining:
            train_surrogate()
        optimize()    
else:
    print("There is nothing to perform within this model")

input("Ended")
