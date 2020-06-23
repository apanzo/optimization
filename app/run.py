"""
This is the main module

Notes:
    * It is assumed one problem is run at a time
"""
# Import custom packages
from core import Model, Surrogate, Optimization
from settings import settings,update_settings

def train_surrogate():
    while not surrogate.trained:
        if not surrogate.retraining:
            surrogate.sample()
            surrogate.evaluate()
        surrogate.load_results()
        surrogate.train()
        surrogate.surrogate_convergence()

def optimize():
    # Solve the optimiaztion problem
    optimization.optimize()

    # Verify whether the optimization result agrees with original model
    optimization.verify()
        
# Choose problem to solve
problem_id = 6

# Initialize the settings
update_settings(problem_id)

# Initialize the model
model = Model() 

surrogate = Surrogate(model)
train_surrogate()

##optimization = Optimization(model,surrogate.surrogate.predict_values,surrogate.data.range_out,surrogate)
##optimize()

### Surrogate training loop
##while not model.optimization_converged:
##
##    # Train the surrogate
##    if settings["surrogate"]["surrogate"]:
##        train_surrogate()
##
##    # Optimize
##    if settings["optimization"]["optimize"]:
##        optimize()
##    else:
##        break

input("Ended")

