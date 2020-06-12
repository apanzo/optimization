"""
Data handling module.

The aim of the datamod package is to handle the data
"""
# Import native packages
import json
import os

# Import pypi packages
import numpy as np
from pymoo.factory import get_problem

# Import custom packages
from datamod.problems import problems
from datamod.sampling import sample, sample_adaptive
from settings import settings, load_json

adaptive_methods = load_json(os.path.join(settings["root"],"app","config","dataconf","adaptive"))
    
class get_data:
    """
    Import data from an external file.

    """
    
    def __init__(self,file):
        """
        Constructor method.

        Arguments:
            file: string of the resilts file path

        Attibutes:
            dim_in: number of input dimensions
            col_names: names of columns
            dim_out: number of output dimensions
            coordinates: samples coordinates
            response: sample response
            
        """
        self.dim_in = int(np.loadtxt(file,max_rows=1))
        self.col_names = np.loadtxt(file,skiprows=1,max_rows=1,dtype=str,delimiter=",")
        data = np.loadtxt(file,skiprows=2)
        if len(data.shape) < 2:
            data = np.reshape(data,(-1, len(data)))
        self.dim_out = data.shape[1]-self.dim_in
        self.coordinates = data[:,:self.dim_in]
        self.response = data[:,self.dim_in:]
        self.input, self.range_in = normalize(self.coordinates)
        self.output, self.range_out = normalize(self.response)


def resample(points_new,points_now,sampling,dim_in,range_in):
    """
    Determine the coordinates of the new sample.

    Arguments:
        points_new: number of new samples
        points_now: number of current samples
        sampling: sampling strategy
        dim_in: number of input dimensions
        range_in: range of inputs

    Returns:
        coordinates: coordinates of the new samples
    
    """
    # Sample
    input_sample = sample(sampling,points_now+points_new,dim_in)[points_now:,:] # unit coordinates, only pick those that are new
    coordinates = scale(input_sample,range_in) # full coordinates

    return coordinates

def resample_adaptive(surrogates,data):
    """
    STUFF

    not working for multiple dimensions
    """

    exploration, exploitation = adaptive_methods[settings["data"]["adaptive"]]

    test_sample = sample(settings["data"]["sampling"],settings["data"]["adaptive_sample"],data.dim_in)
    test_pred = [sur.predict_values(test_sample) for sur in surrogates]
    test_np = np.array(test_pred)

    worst_new = sample_adaptive(data,sample,test_sample,exploration,exploitation,test_np)

    return worst_new

def load_problem(name):
    """
    Load a pre-defined benchmark problem.

    Arguments:
        name: problem name

    Returns:
        problem: problem object
        range_in: range of the input
        dim_in: number of output dimensions
        dim_out: number of output dimensions
        n_constr: number of constrains
        
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

def normalize(data):
    """
    Normalize data to [0,1] range.

    Arguments:
        data: data to normalize

    Returns:
        data_norm: normalized data
        ranges: normalization ranges
        s
    """
##    breakpoint()
    data_norm = (data-np.amin(data,0))/np.ptp(data,0)
    ranges = np.stack((np.amin(data,0),np.amax(data,0)),1)
    
    return data_norm, ranges

def scale(data,ranges):
    """
    Scale data from [0,1] range to desired range.

    Arguments:
        data: data to scale
        ranges: normalization ranges

    Returns:
        data_scale: scaled data
    """
    data_scale = data*(ranges[:,1]-ranges[:,0])+ranges[:,0]
    
    return data_scale

def make_results_file(target,dim_in):
    """
    Initialize the results file.

    Arguments:
        target: target file path
        dim_in: number of input dimensions

    Todo:
        * column names
    """
    with open(target, "w") as file:
        file.write(str(dim_in))
        file.write("\n"+"x,y,z"+"\n")
