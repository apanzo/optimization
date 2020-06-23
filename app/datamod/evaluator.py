# Import pypi packages
import numpy as np

# Import custom packages
from datamod.results import write_results

class Evaluator:

    def __init__(self):
        pass

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

    def __init__(self,n_constr):
        self.results = ["F","G"] if n_constr else ["F"]

    def evaluate(self,problem,samples,file,n_constr):

        # Evaluate
        response_all = problem.evaluate(samples,return_values_of=self.results,return_as_dictionary=True) # return values doesn't work - pymoo implementatino problem
        response = np.concatenate([response_all[column] for column in response_all if column in self.results], axis=1)
        
        write_results(file,samples,response)


class EvaluatorANSYS:

    def __init__(self):
        pass

    def evaluate(self):
        pass
    
