"""
Surrogate package.

The aim of the metamod package is to produce and run a surrogate modul
"""
# Import native packages
from copy import copy,deepcopy
import numpy as np

# Import pypi packages
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.model_selection import KFold, ShuffleSplit
from smt.surrogate_models import RBF, KRG, GENN

# Import custom packages
from metamod import preproc ### not used now
from settings import settings
# ANN is imported in set_surrogate only if it is need


def train_surrogates(data,template):
    """
    Train the defined surrogate on the provided data.

    Arguments:
        data: data object
        setting: settings object
        template: surrogate template

    Returns:
        best: best model according to selected metrics
        variance: metrics of surrogate variances for sample size determination
    
    """
    # Unpack settings
    name = settings["surrogate"]["surrogate"]
    validation = settings["surrogate"]["validation"]
    validation_param = settings["surrogate"]["validation_param"]
    
    surrogates = []
    split = set_validation(validation,validation_param)
    
    for train, test in split.split(data.input):
        if template.name == "ANN":
            interp = copy(template)
        else:
            interp = deepcopy(template)
        ### shuffle it!!!
        interp.train_in, interp.train_out = data.input[train], data.output[train]
        interp.test_in, interp.test_out = data.input[test], data.output[test]
        interp.set_training_values(interp.train_in,interp.train_out)
        if interp.name == "ANN":
            interp.set_validation_values(interp.test_in,interp.test_out)

        interp.train()
        interp.ranges = [data.range_in,data.range_out]
        interp.metric = MAE(interp.test_out,interp.predict_values(interp.test_in))  ###  tune surrogate selection logic
        surrogates.append(interp)

    return surrogates

        
def select_best_surrogate(surrogates):

    best = surrogates[np.array([sur.metric for sur in surrogates]).argmin()]
    variance = np.var(np.array([sur.metric for sur in surrogates]))
    
    return best, variance


def set_surrogate(name,dim_in,dim_out,no_points):
    """
    Select the desired surrogate model.

    Arguments:
        name: surrogate type
        dim_in: number of input dimensions
        dim_out: number of output dimension
        no_points: number of sample points

    Returns:
        surrogate: surrogate object

    Raises:
        NameError: if the surrogate is not defined

    Todo:
        * genn - not working
        * initial parameters for rbg, krig
    """
    if name=="ann":
        from metamod.ANN import ANN         ### import only when actually used, its slow due to tensorflow
        surrogate = ANN(no_points=no_points,dims=(dim_in,dim_out))
    elif name=="rbf":
        surrogate = RBF(d0=0.55) ### hard-coded
    elif name=="kriging":
        surrogate = KRG(theta0=[1e2]) ### hard-coded
    elif name=="genn":
        surrogate = GENN()
    else:
        raise NameError('Surrogate not defined, choose "ann","rbf","kriging" or "genn"')
    surrogate.options["print_prediction"] = False
    surrogate.options["print_global"] = False
    
    return surrogate

def set_validation(validation,param):
    """
    Select the desired validation technique model.

    Arguments:
        validation: validation technique
        param: validation technique parameter

    Returns:
        split: indices for the split

    Raises:
        NameError: if the validation technique is not defined
    """
    if validation == "holdout":
        if 0 < param < 1:
            split_ratio = param
            split = ShuffleSplit(1,split_ratio)
        else:
            invalid_param()
    elif validation == "rlt":
        if isinstance(param[0], int) and param[0] != 0 and 0 < param[1] < 1:
            no_repeats = param[0]
            split_ratio = param[1]
            split = ShuffleSplit(no_repeats,split_ratio)  
        else:
            invalid_param()
    elif validation == "kfold":
        if isinstance(param, int) and param != 0:
            no_folds = param
            split = KFold(no_folds)
        else:
            invalid_param()
    else:
         raise NameError('Validation not defined')

    return split

def invalid_param():
    """
    Raise error if validation parameter is invalid

    Notes:

    """
    raise ValueError('Invalid validation parameter')
