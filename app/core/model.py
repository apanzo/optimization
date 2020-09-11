"""
This modules sets up the overall properties of the model.

Attributes:
    evaluators (dict): A dictionary of available evaluators.

"""

# Import custom packages
from core.settings import settings
from datamod.evaluator import EvaluatorANSYSAPDL, EvaluatorANSYSWB, EvaluatorBenchmark, EvaluatorData, RealoadNotAnEvaluator

class Model:
    """
    This is the core class of the framework.

    Attributes:
        evaluator (object): The selected evaluator.
        range_in (np.array): Input parameter allowable ranges.
        dim_in (int): Number of input dimensions.
        dim_out (int): Number of output dimensions.
        n_const (int): Number of constraints.
        n_obj (int): Number of objectives.
        
    Notes:
        The ranges need to be specified if direct evaluation.
    """

    def __init__(self):
        # Obtain problem information 
        self.evaluator = evaluators[settings["data"]["evaluator"]]()
        self.range_in, self.dim_in, self.dim_out, self.n_const = self.evaluator.get_info()
        self.n_obj = self.dim_out - self.n_const
        
evaluators = {"ansys_apdl":EvaluatorANSYSAPDL, "ansys_wb":EvaluatorANSYSWB, "benchmark":EvaluatorBenchmark,
              "data":EvaluatorData, "load":RealoadNotAnEvaluator}
