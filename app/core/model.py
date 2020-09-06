# Import custom packages
from core.settings import settings
from datamod.evaluator import EvaluatorANSYSAPDL, EvaluatorANSYSWB, EvaluatorBenchmark, EvaluatorData, RealoadNotAnEvaluator

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
        self.evaluator = evaluators[settings["data"]["evaluator"]]()
        self.range_in, self.dim_in, self.dim_out, self.n_const = self.evaluator.get_info()
        self.n_obj = self.dim_out - self.n_const
        
evaluators = {"ansys_apdl":EvaluatorANSYSAPDL, "ansys_wb":EvaluatorANSYSWB, "benchmark":EvaluatorBenchmark,
              "data":EvaluatorData, "load":RealoadNotAnEvaluator}
