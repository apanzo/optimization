import numpy as np
import os

from datamod import get_data, load_problem, resample, make_results_file, resample_adaptive
from datamod.evaluator import evaluate_benchmark
from datamod.visual import plot_raw, show_problem, vis_design_space, vis_objective_space
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
            raise ValueError("Evaluator not suppported, choose 'benchmark' or '...'")

        # Deactivate constrains if not set otherwise
        ### Constraints handling not yet tested
        if not settings["optimization"]["constrained"]:
            self.n_const = 0

        # Make response file
        self.folder = os.path.join(settings["root"],"data","temp")
        self.file = os.path.join(self.folder,"results.txt")
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
        self.algorithm, self.termination = set_optimization()
        self.optimization_converged = False

    def sample(self):
        """
        Wrapper function to obtrain the new sample points.

        Note: be careful with geometric, grows fast
        """
        # Determine number of new samples
        if self.no_samples == 0: ################
            sample_points = settings["data"]["default_sample_coef"]*self.dim_in
        else:
            if settings["data"]["resampling"] == "linear":
                sample_points = settings["data"]["resampling_param"]
            elif settings["data"]["resampling"] == "geometric":
                sample_points = int(settings["data"]["resampling_param*self.no_samples"])
            else:
                raise ValueError("Resampling method not suppported, choose 'linear' or 'geometric'")

        if self.no_samples == 0 or not settings["data"]["adaptive"]:
            # Obtain new samples
            self.samples = resample(sample_points,self.no_samples,settings["data"]["sampling"],self.dim_in,self.range_in)
        else:
            # adaptive
            self.samples = resample_adaptive(self.surrogates,self.data)

        # Update sample count
        self.no_samples += sample_points
            
    def evaluate(self):
        """
        Wrapper function to call the evaluted problem solver.

        STUFF
        """
        if settings["data"]["evaluator"] == "benchmark":
            evaluate_benchmark(self.system,self.samples,self.file,self.n_const)
        else:
            raise ValueError("Evaluator not suppported, choose 'benchmark' or '...'")
        
    def load_results(self,verify=False):
        """
        Wrapper function to load the results from the results file

        Arguments:
            verify: whether this is a verification run

        STUFF
        """
        if verify:
            self.verification = get_data(self.file)
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
        
        # Define the problem using the surogate
        self.problem = set_problem(self.surrogate,self.surrogate.ranges,self.n_const)

        # Plot the surrogate response
        if settings["visual"]["show_surrogate"]:
            show_problem(self.problem)
            
    def verify(self):
        """
        Wrapper function to verify the optimized solutions.

        Todo - optimization error logic
        """
        # Verify back the result
        self.file = os.path.join(self.folder,"verify.txt")
        make_results_file(self.file,self.dim_in)

        # Set the optimal solutions as new sample
        self.samples = np.reshape(self.res.X, (-1, self.dim_in))

        # Evaluate the samples andn load the results
        self.evaluate()
        self.load_results(verify=True)

        # Calculate error
        self.optimization_error = (100*(self.verification.response-self.res.F)/self.verification.response)
        if True:
            self.optimization_converged = True 

        ##len([step["delta_f"] for step in model.res.algorithm.display.term.metrics])

    def optimize(self):
        """
        Wrapper function to perform optimization.

        STUFF
        """
        self.res = solve_problem(self.problem,self.algorithm,self.termination)

        # Plot the optimization result in design space
        if settings["visual"]["show_result_des"]:
            vis_design_space(self.res)
            
        # Plot the optimization result in objective space
        if settings["visual"]["show_result_obj"]:
            vis_objective_space(self.res)

