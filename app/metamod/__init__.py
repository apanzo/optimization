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


def train_surrogates(data,iteration):
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
        ranges left for reloading
    
    """
    # Unpack settings
    name = settings["surrogate"]["surrogate"]
    validation = settings["surrogate"]["validation"]
    validation_param = settings["surrogate"]["validation_param"]

    # Initialize training setup    
    surrogates = []
    split = set_validation(validation,validation_param)
    no_splits = split.get_n_splits()

    print(f"###### Training using {name} on {len(data.input)} examples ######")

    # Pretrain ANN
    if name == "ann":
        pretrain = set_surrogate(name,data.dim_in,data.dim_out)
        pretrain.progress = [iteration,1]
        pretrain.pretrain(data.input,data.output,iteration)

    # Train
    for idx, (train, test) in enumerate(split.split(data.input)):
        print(f"### Training model {idx+1}/{no_splits} ###")
        interp = set_surrogate(name,data.dim_in,data.dim_out)
        interp.train_in, interp.train_out = data.input[train], data.output[train]
        interp.test_in, interp.test_out = data.input[test], data.output[test]
        interp.set_training_values(interp.train_in,interp.train_out)
        if name == "ann":
            interp.set_validation_values(interp.test_in,interp.test_out)
            interp.progress = [iteration,idx+1,no_splits]
        interp.train()
        interp.range_out = data.range_out
        interp.metric = evaluate_metrics(interp.test_in,interp.test_out,interp.predict_values,["mae","r2"])
        surrogates.append(interp)

    return surrogates

def set_surrogate(name,dim_in,dim_out):
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
        * nopoints actually not needed
    """
    # Obtain default settings
    setup = load_json(os.path.join(settings["root"],"app","config","metaconf",name))

    # Optionaly update with settinf specified by the input file
    if "setup" in settings["surrogate"].keys():
        setup.update(settings["surrogate"]["setup"])

    # If the ANN surrogate is used, add it to available options
    if name=="ann":
        from metamod.ANN import ANN         ### import only when actually used, its slow due to tensorflow
        available.update({"ann": ANN})
        setup.update({"dims":(dim_in,dim_out)})
##        setup.update({"no_points":no_points,"dims":(dim_in,dim_out)})

    # Load the actual surrogate
    surrogate = available[name](**setup)

    # Disable printing during using
    surrogate.options["print_prediction"] = False
    surrogate.options["print_global"] = False
    
    return surrogate

def reload_info():
    status = load_json(os.path.join(settings["folder"],"status"))
    if status["surrogate_trained"]:
        range_in, dim_in, dim_out, n_const = np.array(status["range_in"]), status["dim_in"], status["dim_out"], status["n_const"]
        return range_in, dim_in, dim_out, n_const
    else:
        raise Exception("There is no surrogate to load")

available = {
    "rbf": RBF,
    "kriging": KRG,
    "genn": GENN}



