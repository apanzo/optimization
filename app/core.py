# Import native packages
import os

# Import pypi packages
import numpy as np

# Import custom packages
from datamod import get_data, load_problem, scale
from datamod.evaluator import EvaluatorBenchmark, EvaluatorANSYS, EvaluatorData
from datamod.problems import Custom
from datamod.results import make_data_file, make_response_files, append_verification_to_database
from datamod.sampling import determine_samples, resample_static, resample_adaptive, complete_grid
from metamod import train_surrogates, reload_info
from metamod.deploy import get_partial_input
from metamod.postproc import check_convergence, select_best_surrogate, verify_results
from optimod import set_optimization, solve_problem
from settings import dump_json, dump_object, load_json, load_object, settings
from visumod import plot_raw, vis_design_space, vis_objective_space, sample_size_convergence, vis_objective_space_pcp
from visumod import compare_surrogate, correlation_heatmap, surrogate_response

class Model:
    """
    This is the core class of the framework
    """

    def __init__(self):
        """
        Constructor.

        Arguments:

        Notes:
            * Need to specify ranges if direct
        """

        # Obtain problem information 
        if settings["surrogate"]["surrogate"] == "load":
            self.range_in, self.dim_in, self.dim_out, self.n_const = reload_info()
        elif settings["data"]["evaluator"] == "benchmark":
            self.system, self.range_in, self.dim_in, self.dim_out, self.n_const = load_problem(settings["data"]["problem"])
            self.evaluator = EvaluatorBenchmark(self.system,self.n_const)
        elif settings["data"]["evaluator"] == "ansys":
            self.evaluator = EvaluatorANSYS()
            self.range_in, self.dim_in, self.dim_out, self.n_const = self.evaluator.get_info()
        elif settings["data"]["evaluator"] == "data":
            self.evaluator = EvaluatorData()
            self.range_in, self.dim_in, self.dim_out, self.n_const = self.evaluator.get_info()
        else:
            raise Exception("Error should have been caught on initialization")

        self.n_obj = self.dim_out - self.n_const
        
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

        # Initialize log
        if not settings["surrogate"]["surrogate"] == "load":
            initial_log = {"surrogate_trained":False,"range_in":self.model.range_in.tolist(),"dim_in":self.model.dim_in,"dim_out":self.model.dim_out,"n_const":self.model.n_const}
            dump_json(os.path.join(settings["folder"],"status"),initial_log )
        
    def sample(self):
        """
        Wrapper function to obtrain the new sample points.

        Note:
            - be careful with geometric, grows fast
            - if non-adaptive sampling is used, adaptive must be set to None
        """

        # Track iteration number
        self.sampling_iterations += 1
        print("----------------------------------------")
        print(f"Iteration {self.sampling_iterations}")

        # Determine number of new samples
        no_new_samples = determine_samples(self.no_samples,self.model.dim_in)
        
        # Obtain new samples
        
        if isinstance(self.model.evaluator,EvaluatorData):
            self.samples = self.model.evaluator.get_samples(self.no_samples,no_new_samples)
        elif self.no_samples == 0 or not settings["data"]["adaptive"]:
            self.samples = resample_static(no_new_samples,self.no_samples,self.model.range_in)
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

        self.model.evaluator.generate_results(self.samples,file,self.sampling_iterations)

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
            plot_raw(self.data,self.sampling_iterations)

    def train(self):
        """
        Wrapper function to (re)train the surrogate.

        STUFF
        """
        # Train the surrogates
        self.surrogates = train_surrogates(self.data,self.sampling_iterations)

        # Select the best model
        self.surrogate = select_best_surrogate(self.surrogates)
##        self.surrogate.range_out = self.data.range_out
        self.surrogate.metric["max_iterations"] = self.sampling_iterations

        # Plot the surrogate response
        compare_surrogate(self.data.input,self.data.output,self.surrogate.predict_values,self.sampling_iterations)
            
    def surrogate_convergence(self):
        criterion = settings["data"]["convergence"]

        self.surrogate_metrics.append(self.surrogate.metric[criterion])

        self.trained = check_convergence(self.surrogate_metrics)

        if self.trained:
            print("Surrogate converged")
            # Plot the sample size convergence
            sample_size_convergence(self.surrogate_metrics,criterion)
            correlation_heatmap(self.surrogate.predict_values,self.model.range_in)

        dump_object("stats",self.surrogate_metrics,self.sampling_iterations)

##    def response_grid(self):
##        self.grid = complete_grid(10,self.model.dim_in)
##        self.response = self.surrogate.predict_values(self.grid)
    def plot_response(self,inputs,output,density=10,**kwargs):
        if len(inputs) > 2:
            raise Exception("Too many input dimensions requested for a plot")

        if not isinstance(output,int):
            raise Exception("Too many output dimensions requested for a plot, should specify an integer")

        if output > self.model.dim_out:
            raise Exception("Output dimension out of bounds")
        
        # Convert to 0 based indexing
        inputs = np.array(inputs) - 1
        output = output - 1

        # Get response        
        input_norm = get_partial_input(density,inputs,self.model.dim_in,self.data.input_abs_max,**kwargs)
        output_norm = self.surrogate.predict_values(input_norm)

        # Unormalize
        input_vec = scale(input_norm,self.model.range_in)
        output_vec = scale(output_norm,self.data.range_out)

        # Make plot
        surrogate_response(input_vec[:,inputs],output_vec[:,[output]],inputs)

    def save(self):
        if self.surrogate.name == 'ANN':
            self.surrogate.model.save(os.path.join(settings["folder"],"logs","ann"))
        else:
            dump_object("meta",self.surrogate)

        status = load_json(os.path.join(settings["folder"],"status"))
        to_update = {"surrogate_trained":True,"range_out":self.surrogate.range_out.tolist()}
        status.update(to_update)
        dump_json(os.path.join(settings["folder"],"status"),status)

    def reload(self):
        status = load_json(os.path.join(settings["folder"],"status"))

        if "ann" in os.listdir(os.path.join(settings["folder"],"logs")):
            from metamod.ANN import ANN ############################################
            from tensorflow.keras.models import load_model as load_keras_model ############################################
            
            interp = ANN()
            interp.nx = status["dim_in"]
            interp.ny = status["dim_out"]
            interp.model = load_keras_model(os.path.join(settings["folder"],"logs","ann"))
            interp.range_out = np.array(status["range_out"])
            interp.options["print_global"] = False
            self.surrogate = interp
        else:
            self.surrogate = load_object("meta")[0]

        
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
                self.error_measure = np.mean(self.error,0)

            print(f"Optimization {measure} percent error: {self.error_measure:.2f}")
            
            if self.error_measure <= settings["optimization"]["error_limit"]:
                self.converged = True
                print("\n############ Optimization finished ############")
                print(f"Total number of samples: {self.surrogate.no_samples:.0f}")

                return

        # If the model has not converged yet
        self.surrogate.trained = False
        self.iterations += 1
        if settings["surrogate"]["append_verification"]:
                self.surrogate.append_verification()

        return

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

