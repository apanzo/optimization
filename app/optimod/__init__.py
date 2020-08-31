"""
Optimization package.

The aim of the optimod package is to perform optimization
"""
# Import native packages
import os

# Import pypi packages
import numpy as np
from pymoo.factory import get_algorithm
from pymoo.factory import get_reference_directions
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.factory import get_termination
from pymoo.optimize import minimize

# Import custom packages
from datamod import scale
from optimod.termination import default_termination
from core.settings import load_json, settings

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

    res_norm = minimize(problem,algorithm,termination,verbose=True)

    # Unnormalize the results
    if direct:
        res = res_norm
    else:
        if res_norm.F is not None:
            res = unnormalize_res(res_norm,problem.norm_in,problem.norm_out)
        else:
            res = None

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
    alg = set_algorithm(setup["algorithm"],no_obj,setup)

    # Get termination criterion
    termination = setup["termination"]

    if termination == "default":
        terimnation = "f_tol"
        setup_term = [0.001,5,5,150,None]
        term = default_termination(no_obj,setup_term)
    else:
        term = get_termination(termination, *setup["termination_val"])

    return alg, term

def unnormalize_res(res,norm_in,norm_out):
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

    res.X = scale(np.atleast_2d(res.X),norm_in)
    res.F = scale(np.atleast_2d(res.F),norm_out[:res.F.shape[-1]]) # slicing to exclude constraints

    return res

def get_operator(name,setup):
    """
    docstring
    """
    
    if name == "mutation":
        operator = get_mutation(**setup["operators"][name])
    elif name == "crossover":
        operator = get_crossover(**setup["operators"][name])
    else:
        raise Exception("Invalid operator requested")

    return operator

def set_algorithm(name,no_obj,setup):
    """
    docstring
    """
    if name == "default":
        if no_obj == 1:
            name = "ga"
        elif no_obj > 1:
            name = "nsga3"

    # Get settings
    setup_gen = load_json(os.path.join(settings["root"],"app","config","opticonf","general"))
    setup_alg = load_json(os.path.join(settings["root"],"app","config","opticonf",name))
    algorithm_args = {}
    
    # Get optimization settings objects
    algorithm_args["sampling"] = get_sampling(setup_gen["sampling"])

    if "operators" in setup:
        for operator in setup["operators"]:
            algorithm_args[operator] = get_operator(operator,setup)

    # Get reference directions
    if name == "nsga3":
        algorithm_args["ref_dirs"] = get_reference_directions("energy", no_obj, setup_alg["ref_dirs_coef"]*no_obj)

    # Modify population
    if "n_offsprings" in setup:
        algorithm_args["n_offsprings"] = setup["n_offsprings"]  
    if "pop_size" in setup:
        algorithm_args["pop_size"] = setup["pop_size"]

    algorithm = get_algorithm(name,eliminate_duplicates=True,**algorithm_args)

    return algorithm
