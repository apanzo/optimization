"""
Optimization package.

The aim of the optimod package is to perform optimization
"""
import os

# Import pypi packages
import numpy as np
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.factory import get_termination
from pymoo.factory import get_algorithm
from pymoo.factory import get_reference_directions
from pymoo.optimize import minimize

# Import custom packages
from datamod import scale
from datamod.problems import Custom
from optimod.termination import default_termination
from settings import load_json, settings

def solve_problem(problem,algorithm,termination,direct):
    """
    Solve the defined problem.

    Arguments:
        problem: problem to solve
        algorithm: optimization algorithm
        termination: termination method
        setting: settings object

    Returns:
        res: results object

    Notes:
        * save_history will work with ANN surrogate from Tensorflow only if line 290 in site-packages\pymoo\model\algorithm is changed from deepcopy to copy
        * optional seed not implemented
    """
    res_nor = minimize(problem,
                   algorithm,
                   termination,
##                   save_history=True,
                   verbose=True)

    if not direct:
        if res_nor.F is not None:
            # Unnormalize the results
            res = unnormalize_res(res_nor,problem.ranges)
        else:
            res = None
    else:
        res = res_nor
    return res

def set_optimization(no_obj):
    """
    Set the selected optimization technique.

    Raises:
        alg: optimization algorithm
        term: termination method

    """
    setup = settings["optimization"]

    # Get optimization algorithm
    name = setup["algorithm"]

    if name == "default":
        if no_obj == 1:
            name = "ga"
        elif no_obj > 1:
            name = "nsga3"

    # Get settings
    setup_alg = load_json(os.path.join(settings["root"],"app","config","opticonf",name))
    setup_gen = load_json(os.path.join(settings["root"],"app","config","opticonf","general"))
    algortihm_args = {}
    
    # Get optimization settings objects
    algortihm_args["sampling"] = get_sampling(setup_gen["sampling"])

    if "operators" in setup:
        for operator in setup["operators"]:
            algortihm_args[operator] = get_operator(operator,setup)

    # Get reference directions
    if name == "nsga3":
        algortihm_args["ref_dirs"] = get_reference_directions("energy", no_obj, setup_alg["ref_dirs_coef"]*no_obj)

    # Modify population
    if "n_offsprings" in setup:
        algortihm_args["n_offsprings"] = setup["n_offsprings"]  
    if "pop_size" in setup:
        algortihm_args["pop_size"] = setup["pop_size"]
        
    alg = get_algorithm(name,eliminate_duplicates=True,**algortihm_args)
    
    # Get termination criterion
    termination = setup["termination"]

    if termination == "default":
        terimnation = "f_tol"
        setup_term = [0.001,5,5,150,None]
        term = default_termination(no_obj,setup_term)
    else:
        term = get_termination(termination, *setup["termination_val"])
    
    return alg, term

def unnormalize_res(res,ranges):
    """
    Unnormliazes results.

    Arguments:
        res: results object (unnormalized)
        ranges: normalization ranges

    Returns:
        res: results object (normalized)

    Implemented for: F,X
    """

    res.X_norm = res.X
    res.F_norm = res.F

    res.X = scale(np.atleast_2d(res.X),ranges[0])
    res.F = scale(np.atleast_2d(res.F),ranges[1][:res.F.shape[-1],:]) # slicing to exclude constraints

    return res

def set_problem(function,ranges,n_obj,n_constr):
    """
    define the problem from Custom class.

    Arguments:
        surrogate: trained surrogate response
        ranges: ranges of the problem
        n_constr: number of constrains

    Returns:
        prob: problem object

    Notes:
        Assumes a [-1,1] range if not specified
    """
    n_var = ranges[0].shape[0]
    if ranges[1] is None:
        ranges = np.array(ranges).T
        prob = Custom(function,ranges[0].T[0],ranges[0].T[1],n_obj,n_constr)
    else:
        prob = Custom(function,[-1]*n_var,[1]*n_var,n_obj,n_constr)
        prob.ranges = ranges
    
    return prob

def get_operator(name,setup):
    """
    docstring
    """
    
    if name == "mutation":
        operator = get_mutation(**setup[operator])
    elif name == "crossover":
        operator = get_crossover(**setup[operator])
    else:
        raise Exception("Invalid operator requested")

    return operator
