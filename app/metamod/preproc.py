"""
This module provides surrogate pre-processing.

preprocessing stuff
"""
# Import pypi packages
from sklearn.model_selection import train_test_split

# Functions
def split(data,split_ratio):
    """
    Split data into 2 dataset, training and testing

    Arguments:
        data: data object
        split_ratio: split ratio

    Returns:
        indices: indices of the split
    """
    indices = train_test_split(data.input,data.output, test_size=split_ratio)

    return indices

# Core

##### PRE-PROCESS
#### determine the number of input features
####n_features = X_train.shape[1]
##n_features = 2
