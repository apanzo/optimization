"""
Module to assist the deployment of the surrogate.
"""
# Import pypi packages
import numpy as np

from datamod.sampling import response_grid

# Functions
def get_plotting_coordinates(density,requested_dims,dim_in,normalization_factors,range_norm,constants):
    """
    Obtain the grid samples for plotting.
    
    Args:
        density (int): Sampling density of the reponse plot.
        requested_dims (list): Input dimensions to plot.
        dim_in (int): Number of input dimensions.
        normalization_factors (np.array): Input normalization factors.
        range_norm (np.array): Range of validity in normalized coordinates.
        constants (list): Values of the fixed input dimensions.

    Returns:
        samples (np.array): Grid samples. 

    """    
    all_samples = get_input_coordinates(density,requested_dims,range_norm)
    
    if len(requested_dims) == dim_in:
        if constants is not None:
            print("Returning full grid, ignoring constants")            
        samples = all_samples
        
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

        samples = partial_input

    return samples

def get_input_coordinates(density,requested_dims,range_norm):
    """
    Obtain the grid samples.

    Args:
        density (int): Sampling density of the reponse plot.
        requested_dims (list): Input dimensions to plot.
        range_norm (np.array): Range of validity in normalized coordinates.
        
    Returns:
        samples (np.array): Grid samples. 
    """
    samples = response_grid(density,requested_dims,range_norm)

    return samples
