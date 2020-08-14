"""
This module provides surrogate pre-processing.

preprocessing stuff
"""
# Import pypi packages
import numpy as np

from datamod.sampling import complete_grid, partial_grid

# Functions
def get_partial_input(density,requested_dims,dim_in,norm_fact,constants=None):

    if len(requested_dims) > dim_in:
        raise Exception("Too many dimensions specified")

    if not max(requested_dims)in range(dim_in) or not min(requested_dims)in range(dim_in):
        raise Exception("Specified dimensions out of bounds")

    if len(requested_dims) == dim_in:
        if constants is not None:
            print("Performing full grid, ignoring constants")
            
        return complete_grid(density,dim_in)
    else:
        if constants is None or not len(constants) + len(requested_dims) == dim_in:
            raise Exception(f"Incorrect amount of constants specified, required {dim_in-len(requested_dims)}")

        constant_dims = [dim for dim in range(dim_in) if dim not in requested_dims]
        div_fact = norm_fact[constant_dims]
        constants_norm = constants/div_fact
        
        partial_input = partial_grid(density,dim_in,requested_dims)

        partial_input[:,constant_dims] += constants_norm

        return partial_input


