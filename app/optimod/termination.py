"""
Module to provide the termination object.
"""
from pymoo.util.termination.f_tol import MultiObjectiveSpaceToleranceTermination as term_ftol
from pymoo.util.termination.f_tol_single import SingleObjectiveSpaceToleranceTermination as term_ftol_single

def default_termination(n_obj,params):
    """
    Text.

    Args:
        n_obj (int): Number of objectives.
        params (list): Termination parameters.

    Returns:
        termination (): Termination object.
    """
    if n_obj == 1:
        term = term_ftol_single(*params)
    else:
        term = term_ftol(*params)

    return term
