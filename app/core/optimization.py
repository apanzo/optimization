# Import native packages
import os

# Import pypi packages
import numpy as np

# Import custom packages
from core.settings import settings
from datamod.problems import Custom
from metamod.performance import verify_results
from optimod import set_optimization, solve_problem
from visumod import vis_design_space, vis_objective_space, vis_objective_space_pcp

class Optimization:
    """
    Docstring
    """
    def __init__(self,model):

        self.model = model

        self.iterations = 0
        self.converged = False

        # Obtain optimization setup
        self.algorithm, self.termination = set_optimization(model.n_obj)
        
        # Deactivate constrains if not set otherwise
        if settings["optimization"]["constrained"]:
            self.n_const = self.model.n_const
        else:
            self.n_const = 0

        self.direct = not bool(settings["surrogate"]["surrogate"])

    def set_problem(self,surrogate):
        """
        Wrapper function to set the problem.

        Notes: direct optimization does not normalize
        """
        print("###### Optimization #######")

        if self.direct:
            # Specify range
            self.range_in = np.array(settings["optimization"]["ranges"])
            self.function = self.model.evaluator.evaluate
        else:
            self.surrogate = surrogate
            self.range_in = self.surrogate.data.range_in
            self.function = self.surrogate.surrogate.predict_values

        self.problem = Custom(self.function,self.range_in.T[0],self.range_in.T[1],self.model.n_obj,self.n_const)

        if not self.direct:
            self.problem.norm_in = self.surrogate.data.norm_in
            self.problem.norm_out = self.surrogate.data.norm_out
        
    def optimize(self):
        """
        Wrapper function to perform optimization.
        """

        self.res = solve_problem(self.problem,self.algorithm,self.termination,self.direct)

        if self.res is not None:
            self.plot_results()

    def plot_results(self):
        # Plot the optimization result in design space
        vis_design_space(self.res.X,self.iterations)
            
        # Plot the optimization result in objective space
        vis_objective_space(self.res.F,self.iterations)
        if self.model.n_obj > 1:
            vis_objective_space_pcp(self.res.F,self.iterations)

    def verify(self):
        """
        Wrapper function to verify the optimized solutions.
        """        
        if self.res is not None:
            # Evaluate randomly selected samples using surrogate
            verificiation_idx = verify_results(self.res.X, self.surrogate)

            # Calculate error
            response_F = self.surrogate.verification.response[:,:-self.problem.n_constr or None]
            self.optimum_model = response_F[-len(verificiation_idx):,:]
            self.optimum_surrogate = self.res.F[verificiation_idx]
            self.error = np.abs((100*(self.optimum_model-self.optimum_surrogate)/self.optimum_model))

            # Evaluate selected measure
            measure = settings["optimization"]["error"]
            if measure == "max":
                self.error_measure = np.max(self.error)
            elif measure == "mean":
                self.error_measure = np.max(np.mean(self.error,0))

            print(f"Optimization {measure} percent error: {self.error_measure:.2f}")

            self.converged = self.error_measure <= settings["optimization"]["error_limit"]

        else:
            self.converged = False
            
        if self.converged:
            print("\n############ Optimization finished ############")
            print(f"Total number of samples: {self.surrogate.no_samples:.0f}")
        else:
            self.surrogate.trained = False
            self.iterations += 1
            if settings["surrogate"]["append_verification"]:
                    self.surrogate.append_verification()

    def report(self):
        path = os.path.join(settings["folder"],"logs",f"optimizatoin_iteration_{self.iterations}.txt")

        with open(path, "a") as file:
            file.write("======= F ========\n")
            np.savetxt(file,self.res.F,fmt='%.6g')
            file.write("\n======= X ========\n")
            np.savetxt(file,self.res.X,fmt='%.6g')
            if not self.direct:
                file.write("\n======= VERIFICATION ========\n")
                stats = np.concatenate((self.optimum_model,self.optimum_surrogate),1)
                np.savetxt(file,stats,fmt='%.6g')
                file.write("\n======= ERROR ========\n")
                np.savetxt(file,self.error,fmt='%.6g')
            file.write("\n")
            
        ##len([step["delta_f"] for step in model.res.algorithm.display.term.metrics])
