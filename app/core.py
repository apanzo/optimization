# Import native packages
import os

# Import pypi packages
import numpy as np

# Import custom packages
from datamod import get_data, load_problem
from datamod.evaluator import EvaluatorBenchmark, EvaluatorANSYS
from datamod.results import make_data_file, make_response_files, append_verification_to_database
from datamod.sampling import determine_samples, resample_static, resample_adaptive
from metamod import train_surrogates, select_best_surrogate, check_convergence, verify_results
from optimod import set_optimization, set_problem, solve_problem
from settings import settings
from visumod import plot_raw, show_problem, vis_design_space, vis_objective_space, sample_size_convergence

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
        """

        # Obtain problem information 
        if settings["data"]["evaluator"] == "benchmark":
            self.system, self.range_in, self.dim_in, self.dim_out, self.n_const = load_problem(settings["data"]["problem"])
            self.evaluator = EvaluatorBenchmark(self.system,self.n_const)
        elif settings["data"]["evaluator"] == "ansys":
            self.evaluator = EvaluatorANSYS()
            self.range_in, self.dim_in, self.dim_out, self.n_const = self.evaluator.get_info()
        else:
            raise Exception("Error should have been caught on initialization")

        self.n_obj = self.dim_out - self.n_const

        # Need to specify ranges if direct
        
class Surrogate:
    
    def __init__(self,model):

        # Carry over model
        self.model = model

        # Training status
        self.trained = False
        
        # Initialize sampling
        self.no_samples = 0
        self.sampling_iterations = 0

        # Initialize metrics
        self.surrogate_metrics = []

        # Make response files
        self.file, self.verification_file = make_response_files(settings["folder"],self.model.dim_in,self.model.n_obj,self.model.n_const)

    def sample(self):
        """
        Wrapper function to obtrain the new sample points.

        Note: be careful with geometric, grows fast
        """

        # Track iteration number
        self.sampling_iterations += 1
        print("----------------------------------------")
        print(f"Iteration {self.sampling_iterations}")

        # Determine number of new samples
        no_new_samples = determine_samples(self.no_samples,self.model.dim_in)
        
        # Obtain new samples
        if self.no_samples == 0 or not settings["data"]["adaptive"]: ## So if non-adaptive sampling is used, adaptive must be set to None
            self.samples = resample_static(no_new_samples,self.no_samples,self.model.dim_in,self.model.range_in)
        else:
            self.samples = resample_adaptive(no_new_samples,self.surrogates,self.data)

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

        self.model.evaluator.generate_results(self.samples,file)

    def append_verification(self):
        # Add verification results to database
##            if self.retraining and settings["surrogate"]["append_verification"]:
        append_verification_to_database(self.file,self.verification_file)
        
    def load_results(self,verify=False):
        """
        Wrapper function to load the results from the results file

        Arguments:
            verify: whether this is a verification run

        STUFFs
        """
        if verify:
            self.verification = get_data(self.verification_file)
        else:
            # Read database
            self.data = get_data(self.file)
            # Plot the input data

            plot_raw(self.data,"scatter")

    def train(self):
        """
        Wrapper function to (re)train the surrogate.

        STUFF
        """
        # Train the surrogates
        self.surrogates = train_surrogates(self.data)

        # Select the best model
        self.surrogate = select_best_surrogate(self.surrogates)
        self.surrogate.metric["max_iterations"] = self.sampling_iterations

        # Plot the surrogate response
##        show_problem(self.problem)
            
    def surrogate_convergence(self):
        criterion = settings["data"]["convergence"]

        self.surrogate_metrics.append(self.surrogate.metric[criterion])

        self.trained = check_convergence(self.surrogate_metrics)

        if self.trained:
            print("Surrogate converged")
            # Plot the sample size convergence
            sample_size_convergence(self.surrogate_metrics,criterion)
            ##compare()
        
class Optimization:
    """
    Docstring
    """
    def __init__(self,model,surrogate):

        self.model = model
        self.surrogate = surrogate

        # Obtain optimization setup
        self.algorithm, self.termination = set_optimization()
        # Deactivate constrains if not set otherwise
        if not settings["optimization"]["constrained"]:
            self.n_const = 0

        self.direct = not bool(settings["surrogate"]["surrogate"])
        if self.direct:
            # Specify range
            self.ranges = [np.array(settings["optimization"]["ranges"]),None]
            self.function = self.model.evaluator.evaluate
        else:
            self.ranges = [surrogate.data.range_in,surrogate.data.range_out]
            self.function = self.surrogate.surrogate.predict_values

        self.converged = False
            
    def optimize(self):
        """
        Wrapper function to perform optimization.

        STUFF
        """
        print("###### Optimization #######")

        # Define the problem using the surogate
##        if settings["surrogate"]["surrogate"]:
##            self.problem = set_problem(self.function,self.ranges,self.model.n_obj,self.model.n_const)
##        else:
##            self.problem = set_problem(self.model.evaluator.evaluate,self.ranges,self.model.n_obj,self.model.n_const)

        self.problem = set_problem(self.function,self.ranges,self.model.n_obj,self.model.n_const)

        self.res = solve_problem(self.problem,self.algorithm,self.termination,self.direct)

        if self.res is not None: 
            # Plot the optimization result in design space
            vis_design_space(self.res)
                
            # Plot the optimization result in objective space
            vis_objective_space(self.res)        
    
    def verify(self):
        """
        Wrapper function to verify the optimized solutions.

        Todo - optimization error logic
        """
        
        if self.res is not None:        
            idx = verify_results(self.res.X, self.surrogate)

            # Calculate error
            response_F = self.surrogate.verification.response[:,:-self.problem.n_constr or None][-len(idx):,:]
            self.error = (100*(response_F-self.res.F[idx])/response_F)

            self.error_max = np.max(np.abs(self.error))
            self.error_mean = np.mean(self.error,0)

            print("Maximal percentage optimization error: %.2f " % (self.error_max))
            
            if self.error_max < settings["optimization"]["error_limit"]:
                self.converged = True
                print("\n############ Optimization finished ############")
                print("Total number of samples: %.0f" %(self.surrogate.no_samples))
            else:
                self.surrogate.trained = False

##            return idx
        else:
            self.surrogate.trained = False

##            return None

        ##len([step["delta_f"] for step in model.res.algorithm.display.term.metrics])
