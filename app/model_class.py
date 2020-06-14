# Import native packages
import os
from pathlib import Path

# Import pypi packages
import numpy as np

# Import custom packages
from datamod import get_data, load_problem, resample, make_results_file, resample_adaptive
from datamod.evaluator import evaluate_benchmark
from datamod.visual import plot_raw, show_problem, vis_design_space, vis_objective_space, ss_convergence
from metamod import set_surrogate, train_surrogates, select_best_surrogate
from optimod import set_optimization, set_problem, solve_problem
from settings import settings

class Model:
    """
    This is the core class of the framework
    """

    def __init__(self):
        """
        Constructor.

        Arguments:

        Todo:
            * real problem information
            * constraint handling
            * target file ID
        """

        # Obtain problem information 
        if settings["data"]["evaluator"] == "benchmark":
             self.system, self.range_in, self.dim_in, self.dim_out, self.n_const = load_problem(settings["data"]["problem"])
        else:
            raise Exception("Error should have been caught on initialization")

        # Deactivate constrains if not set otherwise
        ### Constraints handling not yet tested
        if not settings["optimization"]["constrained"]:
            self.n_const = 0

        # Make response file
        folder_name = str(settings["data"]["id"]).zfill(2) + "-" +  settings["data"]["problem"]
        self.folder = os.path.join(settings["root"],"data",folder_name)
        Path(self.folder).mkdir(parents=True,exist_ok=True) # parents in fact not needed
        self.file = os.path.join(self.folder,"database.txt")
        make_results_file(self.file,self.dim_in)

        # Initialize sampling
        self.no_samples = 0
        self.sampling_iterations = 0
        self.tracking = [self.no_samples] ## so that object can be passed to set_surrogate instead of number
        self.trained = False
        self.surrogate_metrics = []

        # Load surrogate
        self.surrogate_template = set_surrogate(settings["surrogate"]["surrogate"],self.dim_in,self.dim_out,self.tracking[0])

        # Obtain optimization setup
        self.optimization_converged = False
        if settings["optimization"]["optimize"]:
            self.algorithm, self.termination = set_optimization()
            

    def sample(self):
        """
        Wrapper function to obtrain the new sample points.

        Note: be careful with geometric, grows fast
        """
        # Determine number of new samples
        if self.no_samples == 0:
            no_new_samples = settings["data"]["default_sample_coef"]*self.dim_in
        else:
            if settings["data"]["resampling"] == "linear":
                no_new_samples = settings["data"]["resampling_param"]*self.dim_in
            elif settings["data"]["resampling"] == "geometric":
                no_new_samples  = int(settings["data"]["resampling_param"]*self.no_samples)
            else:
                raise Exception("Error should have been caught on initialization")

        # Obtain new samples
        if self.no_samples == 0 or not settings["data"]["adaptive"]: ## So if non-adaptive sampling is used, adaptive must be set to None
            self.samples = resample(no_new_samples,self.no_samples,settings["data"]["sampling"],self.dim_in,self.range_in)
        else: # if the sampling is adaptive
            self.samples = resample_adaptive(self.surrogates,self.data)

        # Update sample count
        self.no_samples += no_new_samples
            
    def evaluate(self,verify=False):
        """
        Wrapper function to call the evaluted problem solver.

        STUFF
        """
        if verify:
            file = self.verification_file
        else:
            file = self.file
            
        if settings["data"]["evaluator"] == "benchmark":
            evaluate_benchmark(self.system,self.samples,file,self.n_const)
        else:
            raise Exception("Error should have been caught on initialization")
        
    def load_results(self,verify=False):
        """
        Wrapper function to load the results from the results file

        Arguments:
            verify: whether this is a verification run

        STUFF
        """
        if verify:
            self.verification = get_data(self.verification_file)
        else:
            self.data = get_data(self.file)
            # Plot the input data
            if settings["visual"]["show_raw"]:
                plot_raw(self.data,"scatter")

    def train(self):
        """
        Wrapper function to (re)train the surrogate.

        STUFF
        """
        # Train the surrogates
        self.surrogates = train_surrogates(self.data,self.surrogate_template)

        # Select the best model
        self.surrogate, metric = select_best_surrogate(self.surrogates)

        # Store the metric for sample size determination
        self.surrogate_metrics.append(metric)
        
        # Plot the surrogate response
        if settings["visual"]["show_surrogate"]:
            show_problem(self.problem)
            
    def verify(self):
        """
        Wrapper function to verify the optimized solutions.

        Todo - optimization error logic
        """
        if self.res is not None:        
            # Verify back the result
            self.verification_file = os.path.join(self.folder,"verification.txt")
            make_results_file(self.verification_file,self.dim_in)

            # Set the optimal solutions as new sample
            self.samples = np.reshape(self.res.X, (-1, self.dim_in))

            # Evaluate the samples and load the results
            self.evaluate(verify=True)
            self.load_results(verify=True)

            # Calculate error
            response_F = self.verification.response[:,:-self.problem.n_constr]
            self.optimization_error = (100*(response_F-self.res.F)/response_F)

            self.optimization_error_max = np.max(np.abs(self.optimization_error))
            self.optimization_error_mean = np.mean(self.optimization_error,0)

            print(self.optimization_error_max)
            
            if self.optimization_error_max < settings["optimization"]["error_limit"]:
                self.optimization_converged = True
            else:
                self.trained = False
        else:
            self.trained = False

        ##len([step["delta_f"] for step in model.res.algorithm.display.term.metrics])

    def optimize(self):
        """
        Wrapper function to perform optimization.

        STUFF
        """
        # Define the problem using the surogate
        self.problem = set_problem(self.surrogate,self.surrogate.ranges,self.n_const)

        self.res = solve_problem(self.problem,self.algorithm,self.termination)

        if self.res is not None: 
            # Plot the optimization result in design space
            if settings["visual"]["show_result_des"]:
                vis_design_space(self.res)
                
            # Plot the optimization result in objective space
            if settings["visual"]["show_result_obj"]:
                vis_objective_space(self.res)

    def surrogate_convergence(self):
        if settings["surrogate"]["convergence"] == "max_iterations":
            if self.sampling_iterations >= settings["surrogate"]["convergence_limit"]:
                self.trained = True
                print("Surrogate converged")
                if settings["visual"]["show_convergence"]: 
                    # Plot the sample size convergence
                    ss_convergence(self)
                ##compare()
        else:
            raise Exception("Error should have been caught on initialization")

