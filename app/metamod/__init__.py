"""
Surrogate package.

The aim of the metamod package is to produce and run a surrogate modul
"""
# Import native packages
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Import pypi packages
from smt.surrogate_models import RBF, KRG

# Import custom packages
from metamod.performance import evaluate_metrics
from metamod.validation import set_validation
from core.settings import load_json, settings
# ANN is imported in set_surrogate only if it is need
from metamod.ANN_pt import ANN_pt

##import ctypes
##hllDll = ctypes.WinDLL("C:\\Users\\antonin.panzo\\Downloads\\cudart64\\cudart64_100.dll")


def optimize_hyperparameters(data,iteration):
    """
    Train the defined surrogate on the provided data.
    """
    name = settings["surrogate"]["surrogate"]

    model = set_surrogate(name,data.dim_in,data.dim_out)
##    model.progress = [iteration,1]
    model.pretrain(data.input,data.output,iteration)

    return True

def cross_validate(data,iteration):
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

    # Initialize training setup    
    surrogates = []
    split = set_validation(validation,validation_param)
    no_splits = split.get_n_splits()

    print(f"###### Training using {name} on {len(data.input)} examples ######")

    # Train
    for idx, (train, test) in enumerate(split.split(data.input)):
        print(f"### Training model {idx+1}/{no_splits} ###")
        model = set_surrogate(name,data.dim_in,data.dim_out)
        model.train_in, model.train_out = data.input[train], data.output[train]
        model.test_in, model.test_out = data.input[test], data.output[test]
        model.set_training_values(model.train_in,model.train_out)
        if name == "ann" or name == "ann_pt":
            model.set_validation_values(model.test_in,model.test_out)
            model.progress = [iteration,idx+1,no_splits]
            model.CV = True
        model.train()
##        model.range_out = data.range_out
        model.metric = evaluate_metrics(model.test_in,model.test_out,model.predict_values,["mae","r2"])
        model.metric["max_iterations"] = iteration
        surrogates.append(model)

    if name == "ann" or name == "ann_pt":
        settings["surrogate"]["early_stop"] = int(np.mean([ann.early_stop for ann in surrogates]))

    return surrogates

def train_surrogate(data):
    name = settings["surrogate"]["surrogate"]

    model = set_surrogate(name,data.dim_in,data.dim_out)
    model.set_training_values(data.input,data.output)
    if name == "ann" or name == "ann_pt":
        model.CV = False
    model.train()

    return model

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
    """
    # Obtain default settings
    setup = load_json(os.path.join(settings["root"],"app","config","metaconf",name))

    # Optionaly update with setting specified by the input file
    if "setup" in settings["surrogate"].keys():
        setup.update(settings["surrogate"]["setup"])

    # If the ANN surrogate is used, add it to available options
    if name=="ann":
        from metamod.ANN import ANN         ### import only when actually used, its slow due to tensorflow
        available.update({"ann": ANN})
        setup.update({"dims":(dim_in,dim_out)})

    if name=="ann_pt":
        setup.update({"dims":(dim_in,dim_out)})

    # Load the actual surrogate
    surrogate = available[name](**setup)

    # Disable printing during using
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
    "ann_pt": ANN_pt}



