"""
Surrogate package.

The aim of the metamod package is to produce and run a surrogate modul
"""
# Import native packages
import numpy as np
import os

# Import pypi packages
from smt.surrogate_models import RBF, KRG, GENN

# Import custom packages
from metamod.postproc import check_convergence, select_best_surrogate, verify_results, evaluate_metrics
from metamod.preproc import set_validation
from settings import load_json, settings
# ANN is imported in set_surrogate only if it is need


def train_surrogates(data):
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
        for pretraining, holout with 0.2 assumed
    
    """
    # Unpack settings
    name = settings["surrogate"]["surrogate"]
    validation = settings["surrogate"]["validation"]
    validation_param = settings["surrogate"]["validation_param"]
    
    surrogates = []
    split = set_validation(validation,validation_param)
    no_points = data.input.shape[0]

    print(f"Training using {name} on {len(data.input)} examples")

    # Pretrain
    if name == "ann":
        pretrain_split = set_validation("holdout",0.2)
        train, test = next(pretrain_split.split(data.input))

        pretrain = set_surrogate(name,data.dim_in,data.dim_out,no_points)
        pretrain.train_in, pretrain.train_out = data.input[train], data.output[train]
        pretrain.test_in, pretrain.test_out = data.input[test], data.output[test]
        pretrain.set_training_values(pretrain.train_in, pretrain.train_out)
        pretrain.set_validation_values(pretrain.test_in,pretrain.test_out)
        pretrain.pretrain()
    
    for train, test in split.split(data.input):
        interp = set_surrogate(name,data.dim_in,data.dim_out,no_points)
        interp.train_in, interp.train_out = data.input[train], data.output[train]
        interp.test_in, interp.test_out = data.input[test], data.output[test]
        interp.set_training_values(interp.train_in,interp.train_out)
        if name == "ann":
            interp.set_validation_values(interp.test_in,interp.test_out)
        interp.train()
##        interp.ranges = [data.range_in,data.range_out]
        interp.metric = evaluate_metrics(interp.test_in,interp.test_out,interp.predict_values,["mae","r2"])
        surrogates.append(interp)

    return surrogates

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
    setup = load_json(os.path.join(settings["root"],"app","config","metaconf",name))
    if "setup" in settings["surrogate"].keys():
        setup.update(settings["surrogate"]["setup"])

    if name=="ann":
        from metamod.ANN import ANN         ### import only when actually used, its slow due to tensorflow
        available.update({"ann": ANN})
        setup.update({"no_points":no_points,"dims":(dim_in,dim_out)})

    surrogate = available[name](**setup)

    surrogate.options["print_prediction"] = False
    surrogate.options["print_global"] = False
    
    return surrogate

available = {
    "rbf": RBF,
    "kriging": KRG,
    "genn": GENN}



