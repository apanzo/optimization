"""
This module provides cross-validation splitting techniques.

Attributes:
    split_methods (dict): CV validation techniques.
"""
# Import pypi packages
from sklearn.model_selection import KFold, ShuffleSplit

# Functions
def set_validation(validation,param):
    """
    Select the desired validation technique model.

    Arguments:
        validation (str): Validation technique name.
        param (float/int): Validation technique parameter.

    Returns:
        split (np.array): Indices for the split.

    """
    split = split_methods[validation](param)
    
    return split

def split_holdout(param):
    """
    Select the desired validation technique model.

    Arguments:
        param (float): Holdout ratio.

    Returns:
        split (np.array): Indices for the split.
    """
    if 0 < param < 1:
        split_ratio = param
        split = ShuffleSplit(1,split_ratio)
    else:
        invalid_param()

    return split

def split_rlt(param):
    """
    Select the desired validation technique model.

    Arguments:
        param (float): Holdout ratio.

    Returns:
        split (np.array): Indices for the split.
    """
    if isinstance(param[0], int) and param[0] != 0 and 0 < param[1] < 1:
        no_repeats = param[0]
        split_ratio = param[1]
        split = ShuffleSplit(no_repeats,split_ratio)  
    else:
        invalid_param()

    return split

def split_kfold(param):
    """
    Select the desired validation technique model.

    Arguments:
        param (int): Number of splits.

    Returns:
        split (np.array): Indices for the split.
    """
    if isinstance(param, int) and param != 0:
        no_folds = param
        split = KFold(no_folds, shuffle=True)
    else:
        invalid_param()

    return split

def invalid_param():
    """
    Raise error if validation parameter is invalid.

    Raises:
        NameError: if validation parameter is invalid.
    """
    raise ValueError('Invalid validation parameter')

split_methods = {"holdout":split_holdout,"rlt":split_rlt,"kfold":split_kfold}
