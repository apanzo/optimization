# Import pypi packages
import numpy as np

def write_result(file,inputs,outputs):
    all_data = np.concatenate((inputs,outputs),1)

    with open(file, 'a') as f:
        lines = ["\t".join([str(i) for i in line]) for line in all_data]
        f.write("\n".join(lines))
        f.write("\n")
        ### perhaps write with numpy
    

def evaluate_benchmark(problem,samples,file,n_constr):    
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

    # Evaluate
    return_values = ["F","G"] if n_constr else ["F"]
    response_dir = problem.evaluate(samples,return_values_of=return_values,return_as_dictionary=True) # return values doesn't work - pymoo implementatino problem
    response = np.concatenate([response_dir[column] for column in response_dir if column in return_values], 1)
    
    write_results(file,samples,outputs)
