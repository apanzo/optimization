"""
Data handling module.

The aim of the datamod package is to handle the data.
"""

# Import pypi packages
import numpy as np
from pymoo.factory import get_problem
from sklearn.preprocessing import normalize as sk_normalize

# Import custom packages
from datamod.problems import problems
from datamod.results import load_results
    
class get_data:
    """
    Import data from an external file.

    Attributes:
        dim_in (int): Number of input dimensions.
        col_names (list): Names of columns.
        dim_out (int): Number of output dimensions.
        coordinates (np.array): Samples coordinates.
        response (np.array): Sample response.
        input (np.array): Normalized input samples.
        output (np.array): Normalized output samples.
        norm_in (np.array): Input normalization factors.
        norm_out (np.array): Output normalization factors.
        range_in (np.array): Range of the input data.
        range_out (np.array): Range of the output data.
    """
    
    def __init__(self,file):
        """
        Args:
            file (str): Path and name of the database file.
        """
        # Load results
        self.dim_in, self.col_names, data = load_results(file)
        if len(data.shape) == 1:
            data = np.atleast_2d(data)
        elif len(data.shape) == 0:
            raise Exception("0 dimensions data")
        elif len(data.shape) > 2:
            raise Exception("Too many data dimensions")
        self.dim_out = data.shape[1]-self.dim_in

        # Separate inputs and outputs
        self.coordinates = data[:,:self.dim_in]
        self.response = data[:,self.dim_in:]

        # Normalize data
        self.input, self.norm_in = normalize(self.coordinates)
        self.output, self.norm_out = normalize(self.response)
        self.range_in = get_range(self.input)
        self.range_out = get_range(self.output)

def load_problem(name):
    """
    Load a pre-defined benchmark problem.

    Arguments:
        name (str): Name of the desired problem.

    Returns:
        problem (): Benchmark problem.
        range_in (np.array): Input parameter allowable ranges.
        dim_in (int): Number of input dimensions.
        dim_out (int): Number of output dimensions.
        n_constr (int): Number of constraints.        
    """

    # Own defined functions
    if name in problems.keys():
        problem = problems[name]()
    # Functions from Pymoo
    else:
        problem = get_problem(name)

    # Process it
    range_in = np.stack((problem.xl,problem.xu),axis=1)
    dim_in = problem.n_var
    n_constr = problem.n_constr
    dim_out = problem.n_obj + problem.n_constr
    
    return problem, range_in, dim_in, dim_out, n_constr

def scale(data,ranges):
    """
    Scale data from [-1,1] range to original range.

    Arguments:
        data (np.array): Data to scale.
        ranges (np.array): Normalization ranges.

    Returns (np.array):
        data_scale: Scaled data.
    """
    data_scale = data*ranges
    
    return data_scale

def normalize(data):
    """
    Normalize the data to the [-1,1] range.

    Arguments:
        data (np.array): Data to normalize.
    
    Todo:
        Revise the outputs.
    """
    
    return sk_normalize(data, norm='max',axis=0,return_norm=True)

def get_range(data):
    """
    Determine the range of the data.
    
    Arguments:
        data (np.array): Data to analyze.
    
    Returns:
        ranges (np.array): Ranges of the given data.
    """
    lower = np.amin(data,0)
    upper = np.amax(data,0)
    ranges = np.stack((lower,upper),1)

    return ranges
