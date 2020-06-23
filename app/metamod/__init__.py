"""
Surrogate package.

The aim of the metamod package is to produce and run a surrogate modul
"""
# Import native packages
import numpy as np
import os

# Import pypi packages
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import r2_score as R2

from smt.surrogate_models import RBF, KRG, GENN

# Import custom packages
from metamod.postproc import check_convergence, select_best_surrogate, verify_results
from metamod.preproc import set_validation
from settings import load_json, settings
# ANN is imported in set_surrogate only if it is need


def train_surrogates(data,dim_in,dim_out,no_points):
    """
    Train the defined surrogate on the provided data.

    Arguments:
        data: data object
        setting: settings object
        template: surrogate template

    Returns:
        best: best model according to selected metrics
        variance: metrics of surrogate variances for sample size determination

    Note:
        the indices are shuffled
    
    """
    # Unpack settings
    name = settings["surrogate"]["surrogate"]
    validation = settings["surrogate"]["validation"]
    validation_param = settings["surrogate"]["validation_param"]
    
    surrogates = []
    split = set_validation(validation,validation_param)
    
    for train, test in split.split(data.input):

        interp = set_surrogate(settings["surrogate"]["surrogate"],dim_in,dim_out,no_points,len(surrogates))
        np.random.shuffle(train), np.random.shuffle(test) ###????
        interp.train_in, interp.train_out = data.input[train], data.output[train]
        interp.test_in, interp.test_out = data.input[test], data.output[test]
        interp.set_training_values(interp.train_in,interp.train_out)
        interp.train()
        interp.ranges = [data.range_in,data.range_out]
        interp.metric = {}
        interp.metric["mae"] = MAE(interp.test_out,interp.predict_values(interp.test_in))
        interp.metric["r2"] = R2(interp.test_out,interp.predict_values(interp.test_in)) 
        surrogates.append(interp)

    return surrogates

        

def set_surrogate(name,dim_in,dim_out,no_points,keras_optimized):
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
    setup = load_json(os.path.join(settings["root"],"app","config","metaconf",name))
    if name=="ann":
        from metamod.ANN import ANN         ### import only when actually used, its slow due to tensorflow
        surrogate = ANN(setup,keras_optimized,no_points=no_points,dims=(dim_in,dim_out))
    elif name=="rbf":
        surrogate = RBF(**setup) #0.55
    elif name=="kriging":
        surrogate = KRG(**setup) # 1e2
    elif name=="genn":
        surrogate = GENN(**setup)
    else:
        raise NameError('Surrogate not defined, choose "ann","rbf","kriging" or "genn"')
    surrogate.options["print_prediction"] = False
    surrogate.options["print_global"] = False
    
    return surrogate



