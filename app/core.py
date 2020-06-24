# Import native packages
import os
from pathlib import Path

# Import pypi packages
import numpy as np

# Import custom packages
from datamod import get_data, load_problem
from datamod.evaluator import EvaluatorBenchmark
from datamod.results import make_data_file, append_verification_to_database
from datamod.sampling import determine_samples, resample_static, resample_adaptive
from datamod.visual import plot_raw, show_problem, vis_design_space, vis_objective_space, sample_size_convergence
from metamod import set_surrogate, train_surrogates, select_best_surrogate, check_convergence, verify_results
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
        """

        # Make workfolder
        self.folder = make_workfolder()

        # Obtain problem information 
        if settings["data"]["evaluator"] == "benchmark":
##             self.system, self.range_in, self.dim_in, self.dim_out, self.n_const = load_problem(settings["data"]["problem"])
            self.system, self.range_in, self.dim_in, self.dim_out, self.n_const = load_problem(settings["data"]["problem"])
            self.evaluator = EvaluatorBenchmark(self.system,self.n_const)
        else:
            raise Exception("Error should have been caught on initialization")

        self.n_obj = self.dim_out - self.n_const

        self.converged = False

class Surrogate:
    
    def __init__(self,model):

        # Carry over model
        self.model = model

        # Training status
        self.trained = False
        self.retraining = False
        
        # Initialize sampling
        self.no_samples = 0
        self.sampling_iterations = 0

        # Initialize metrics
        self.surrogate_metrics = []

        # Make response filess
        self.file = os.path.join(self.model.folder,"database.csv")
        make_data_file(self.file,self.model.dim_in)
        self.verification_file = os.path.join(self.model.folder,"verification.csv")
        make_data_file(self.verification_file,self.model.dim_in)

    def sample(self):
        """
        Wrapper function to obtrain the new sample points.

        Note: be careful with geometric, grows fast
        """

        # Track iteration number
        self.sampling_iterations += 1
        print("----------------------------------------")
        print("Iteration "+str(self.sampling_iterations))

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
            
        if settings["data"]["evaluator"] == "benchmark":
            self.model.evaluator.generate_results(self.samples,file)
        else:
            raise Exception("Error should have been caught on initialization")
        
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
            # Add verification results to database
            if self.retraining:
                append_verification_to_database()
                self.retraining = False

            # Read database
            self.data = get_data(self.file)
            # Plot the input data
            if "raw_data" in settings["visual"]:
                plot_raw(self.data,"scatter")

    def train(self):
        """
        Wrapper function to (re)train the surrogate.

        STUFF
        """
        # Train the surrogates
        self.surrogates = train_surrogates(self.data,self.model.dim_in,self.model.dim_out,self.no_samples)

        # Select the best model
        self.surrogate = select_best_surrogate(self.surrogates)

        # Plot the surrogate response
        if "surrogate" in settings["visual"]:
            show_problem(self.problem)
            
    def surrogate_convergence(self):
        self.trained = check_convergence(self.sampling_iterations,self.surrogate.metric[settings["data"]["convergence"]])

        print(f"Sample size convergence metric: {settings['data']['convergence']} - {self.surrogate.metric[settings['data']['convergence']]}")
        self.surrogate_metrics.append(self.surrogate.metric[settings["data"]["convergence"]])

        if self.trained:
            print("Surrogate converged")
            if "convergence" in settings["visual"]: 
                # Plot the sample size convergence
                sample_size_convergence(self.surrogate_metrics,settings["data"]["convergence"])
            ##compare()
        
class Optimization:
    def __init__(self,model,function,range_out,surrogate,direct=False):

        # Obtain optimization setup
        self.optimization_converged = False
        self.algorithm, self.termination = set_optimization()
        # Deactivate constrains if not set otherwise
        if not settings["optimization"]["constrained"]:
            self.n_const = 0

        self.range_out = range_out
        self.model = model
        self.function = function
        self.surrogate = surrogate
        self.direct = direct

    def optimize(self):
        """
        Wrapper function to perform optimization.

        STUFF
        """
        print("###### Optimization #######")

        # Define the problem using the surogate
        ranges = [self.range_out]

        if settings["surrogate"]["surrogate"]:
            self.problem = set_problem(self.function,self.model.range_in,self.model.n_obj,self.model.n_const)
        else:
            self.problem = set_problem(self.model.evaluator.evaluate,self.model.range_in,self.model.n_obj,self.model.n_const)

        self.res = solve_problem(self.problem,self.algorithm,self.termination,self.direct)

        if self.res is not None: 
            # Plot the optimization result in design space
            if "design_space" in settings["visual"]:
                vis_design_space(self.res)
                
            # Plot the optimization result in objective space
            if "objective_space" in settings["visual"]:
                vis_objective_space(self.res)        
    
    def verify(self):
        """
        Wrapper function to verify the optimized solutions.

        Todo - optimization error logic
        """
        if not settings["surrogate"]["surrogate"]:
            self.optimization_converged = True

            return

        if self.res is not None:        
            idx = verify_results(self.res.X, self.surrogate)
            
            # Calculate error
            breakpoint()
            response_F = self.surrogate.verification.response[:,:-self.problem.n_constr or None]
            self.optimization_error = (100*(response_F-self.res.F[idx])/response_F)

            self.optimization_error_max = np.max(np.abs(self.optimization_error))
            self.optimization_error_mean = np.mean(self.optimization_error,0)

            print("Maximal percentage optimization error: %.2f " % (self.optimization_error_max))
            
            if self.optimization_error_max < settings["optimization"]["error_limit"]:
                self.optimization_converged = True
                print("\n############ Optimization finished ############")
                print("Total number of samples: %.0f" %(self.no_samples))
            else:
                self.trained = False
                self.retraining = True

            return idx
        else:
            self.trained = False
            self.retraining = True

            return None

        ##len([step["delta_f"] for step in model.res.algorithm.display.term.metrics])



def make_workfolder():
    """
    Initialize the workdirectory.

    """
    # Setup the folder path
    folder_name = str(settings["data"]["id"]).zfill(2) + "-" +  settings["data"]["problem"]
    folder_path = os.path.join(settings["root"],"data",folder_name)

    # Create folder, if not done yet
    Path(folder_path).mkdir(parents=True,exist_ok=True) # parents in fact not needed

    return folder_path

