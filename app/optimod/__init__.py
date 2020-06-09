"""
Optimization package.

The aim of the optimod package is to perform optimization
"""
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.factory import get_termination
from pymoo.factory import get_algorithm
from pymoo.optimize import minimize

from datamod.problems import Custom


def solve_problem(problem,algorithm,termination,setting):
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
    opt_seed = setting.opt_seed
    verbose = setting.opt_verbose

    res_nor = minimize(problem,
                   algorithm,
                   termination,
                   seed=opt_seed,
##                   pf=problem.pareto_front(use_cache=False),
##                   save_history=True,
                   verbose=verbose)

    # Unnormalize the results
    res = unnormalize_res(res_nor,problem.ranges)

    return res

def set_optimization(setting):
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
    # Unpack optimization settings
    optimization = setting.optimization
    pop_size = setting.pop_size
    n_offsprings = setting.n_offsprings
    opt_sampling = setting.opt_sampling
    crossover = setting.crossover
    crossover_prob = setting.crossover_prob
    crossover_eta = setting.crossover_eta
    mutation = setting.mutation
    mutation_eta = setting.mutation_eta
    termination = setting.termination
    termination_val = setting.termination_val

    # Get optimization settings objects
    sampling = get_sampling(opt_sampling)
    mutation = get_mutation(mutation, eta=mutation_eta)
    crossover = get_crossover(crossover, prob=crossover_prob, eta=crossover_eta)

    # Get optimization algorithm
    alg = get_algorithm(optimization,
    pop_size=pop_size,
    n_offsprings=n_offsprings,
    sampling=sampling,
    crossover=crossover,
    mutation=mutation,
    eliminate_duplicates=True
)
    # Get termination criterion
    term = get_termination(termination, *termination_val)
    
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

    from datamod import scale
    res.X = scale(res.X,ranges[0])
    res.F = scale(res.F,ranges[1])

    return res

def set_problem(surrogate,ranges,n_constr):
    """
    define the problem from Custom class.

    Arguments:
        surrogate: trained surrogate response
        ranges: ranges of the problem
        n_constr: nu,ber of constrains

    Returns:
        prob: problem object
    """
    n_var = ranges[0].shape[0]
    
    prob = Custom(surrogate,[0]*n_var,[1]*n_var,n_constr=n_constr)
    
    prob.ranges = ranges
    
    return prob
    
