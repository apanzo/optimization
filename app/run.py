"""
This is the main module

Notes:
    * It is assumed one problem is run at a time
"""
# Import custom packages
from core import Model, Surrogate, Optimization
from settings import settings,update_settings

def train_surrogate():
    
    # Train surrorate
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
problem_id = 3

# Initialize the settings
update_settings(problem_id)

# Initialize the model
model = Model() 

build_surrogate = bool(settings["surrogate"]["surrogate"])
perform_optimization = bool(settings["optimization"]["algorithm"])

if build_surrogate and not perform_optimization:
    surrogate = Surrogate(model)
    train_surrogate()
elif perform_optimization and not build_surrogate:
    optimization = Optimization(model,model.system.evaluate,None,None,direct=True)
    optimize()
elif build_surrogate and perform_optimization:
    surrogate = Surrogate(model)
    optimization = Optimization(model,surrogate.surrogate.predict_values,surrogate.data.range_out,surrogate)
    while not model.converged:
        train_surrogate()
        optimize()    
else:
    print("There is nothing to perform within this model")

input("Ended")

