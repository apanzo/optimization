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
from pymoo.optimize import minimize

# Import custom packages
from datamod import scale
from datamod.problems import Custom
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
        save_history will work with ANN surrogate from Tensorflow only if line 290 in site-packages\pymoo\model\algorithm is changed from deepcopy to copy
    """
##    opt_seed = setup_gen["solve"]["opt_seed"]
##    verbose = settings["optimization"]["opt_verbose"]
    res_nor = minimize(problem,
                   algorithm,
                   termination,
##                   seed=opt_seed,
##                   pf=problem.pareto_front(use_cache=False),
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

def set_optimization():
    """
    Set the selected optimization technique.

    Arguments:
        setting: setting object

    Raises:
        alg: optimization algorithm
        term: termination method

    Note:
        there is also a single objective one
        termination_val = [0.0025*1000,5,5,None,None]

    """
    setup = settings["optimization"]
    setup_gen = load_json(os.path.join(settings["root"],"app","config","opticonf",setup["algorithm"]))
    
    # Get optimization settings objects
    sampling = get_sampling(setup_gen["opt_sampling"])
    mutation = get_mutation(**setup_gen["mutation"])
    crossover = get_crossover(**setup_gen["crossover"])

    # Get optimization algorithm

    if setup["algorithm"] == "default":
        raise Exception("Defaults not implemented yet")
        if no_dim == 1:
            raise Exception()
        elif no_dim == 2:
            raise Exception()
        else:
            raise Exception()
    else:
        alg = get_algorithm(setup["algorithm"],
            pop_size=setup["pop_size"],
            n_offsprings=setup["n_offsprings"],
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=True
            )
    
    # Get termination criterion
    term = get_termination(setup["termination"], *setup["termination_val"])
    
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
        Assumes a [-1,1] range
    """
    n_var = ranges[0].shape[0]
    if ranges[1] is None:
        ranges = np.array(ranges).T
        prob = Custom(function,ranges[0].T[0],ranges[0].T[1],n_obj,n_constr)
    else:
        prob = Custom(function,[-1]*n_var,[1]*n_var,n_obj,n_constr)
        prob.ranges = ranges
    
    return prob
   
