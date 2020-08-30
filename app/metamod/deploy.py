"""
This module provides surrogate pre-processing.

preprocessing stuff
"""
# Import pypi packages
import numpy as np

from datamod.sampling import response_grid

# Functions
def get_plotting_coordinates(density,requested_dims,dim_in,normalization_factors,range_norm,constants):
    
    all_samples = get_input_coordinates(density,requested_dims,range_norm)
    
    if len(requested_dims) == dim_in:
        if constants is not None:
            print("Returning full grid, ignoring constants")
            
        return all_samples
    else:
        if constants is None or len(constants) + len(requested_dims) != dim_in:
            raise Exception(f"Incorrect amount of constants specified, required {dim_in-len(requested_dims)}")

        # Obtain full response
        sample = np.zeros((all_samples.shape[0],dim_in))
        sample[:,requested_dims] = all_samples
        partial_input = sample

        # Obtain constant dimensions
        constant_dims = [dim for dim in range(dim_in) if dim not in requested_dims]
        division_factors = normalization_factors[constant_dims]
        constants_norm = constants/division_factors

        # Replace constant dimensions
        partial_input[:,constant_dims] += constants_norm

        return partial_input

def get_input_coordinates(density,requested_dims,range_norm):
    samples = response_grid(density,requested_dims,range_norm)

    return samples
