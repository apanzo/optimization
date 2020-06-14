"""
Optimization package.

The aim of the optimod package is to perform optimization
"""
# Import pypi packages
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.factory import get_termination
from pymoo.factory import get_algorithm
from pymoo.optimize import minimize

# Import custom packages
from datamod import scale
from datamod.problems import Custom
from settings import settings


def solve_problem(problem,algorithm,termination):
    """
    Solve the defined problem.

    Arguments:
        problem: problem to solve
        algorithm: optimization algorithm
        termination: termination method
        setting: settings object

    Returns:
        res: results object
    """
    opt_seed = settings["optimization"]["opt_seed"]
    verbose = settings["optimization"]["opt_verbose"]
    res_nor = minimize(problem,
                   algorithm,
                   termination,
##                   seed=opt_seed,
##                   pf=problem.pareto_front(use_cache=False),
##                   save_history=True,
                   verbose=verbose)

    if res_nor.F is not None:
        # Unnormalize the results
        res = unnormalize_res(res_nor,problem.ranges)
    else:
        res = None

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
    
    # Get optimization settings objects
    sampling = get_sampling(setup["opt_sampling"])
    mutation = get_mutation(setup["mutation"], eta=setup["mutation_eta"])
    crossover = get_crossover(setup["crossover"], prob=setup["crossover_prob"], eta=setup["crossover_eta"])

    # Get optimization algorithm
    alg = get_algorithm(setup["optimization"],
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

    res.X_val = res.X
    res.F_val = res.F
    
    res.X = scale(res.X,ranges[0])
    res.F = scale(res.F,ranges[1][:res.F.shape[-1],:]) # slicing to exclude constraints

    return res

def set_problem(surrogate,ranges,n_constr):
    """
    define the problem from Custom class.

    Arguments:
        surrogate: trained surrogate response
        ranges: ranges of the problem
        n_constr: number of constrains

    Returns:
        prob: problem object
    """
    n_var = ranges[0].shape[0]
    prob = Custom(surrogate,[0]*n_var,[1]*n_var,n_constr=n_constr)
    
    prob.ranges = ranges
    
    return prob
    
