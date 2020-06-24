# Import pypi packages
import numpy as np

# Import custom packages
from datamod.results import write_results

class Evaluator:
    """
    Docstring
    """

    def __init__(self):
        self.save_results = write_results

class EvaluatorBenchmark(Evaluator):
    """
    Evaluate a benchmark problem on the given sample and write the results in a results finle.

    Arguments:
        problem: problem object
        samples: coordinates of new samples
        file: target results file
        n_constr: number of constrains

    Todo:
        * write performance metrics log

    Notes:
        * return values is wrongly implemented in pymoo
    """

    def __init__(self,problem,n_constr):
        super().__init__()
        self.results = ["F","G"] if n_constr else ["F"]
        self.problem = problem

    def evaluate(self,samples):
        # Evaluate
        response_all = self.problem.evaluate(samples,return_values_of=self.results,return_as_dictionary=True) # return values doesn't work - pymoo implementatino problem
        response = np.concatenate([response_all[column] for column in response_all if column in self.results], axis=1)

        return response

    def generate_results(self,samples,file):
        response = self.evaluate(samples)
        self.save_results(file,samples,response)

class EvaluatorANSYS:
    """
    Docstring
    """

    def __init__(self):
        pass

    def evaluate(self):
        pass
    
