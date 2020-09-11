"""
This is the visualization module.
"""

# Import pypi packages
import numpy as np

# Import custom packages
from visumod.plots import scatter,scatter_pymoo,curve,heatmap,pcp,surface_pymoo,learning_curves,pareto_fronts,adaptive_candidates

def plot_raw(data,iteration,normalized=False):
    """
    Plot either a scatter, curve or surface plot.

    Arguments:
        data (np.array): Raw data samples.
        iteration (int): Iteration number.
        normalized (bool): Whether the data is normalized.

    Notes:
        Surface plot not used yet.
    """
    # Select data to plot
    if normalized:
        data_all = np.concatenate((data.input,data.output),1)
    else:
        data_all = np.concatenate((data.coordinates,data.response),1)

##    # Select plot type
##    if name == "scatter":
##        plot_type = None
##    elif name == "surface":
##        plot_type = "other"

    # Save figure
    name = f"iteration_{iteration}_raw"
    scatter_pymoo(data_all,name+"_1","x")

def vis_design_space(data,iteration):
    """
    Visualize the design space in design coordinates.

    Arguments:
        res (pymoo.model.result.Result): Results object.
        iteration (int): Iteration number.
    """
    scatter_pymoo(data,f"optimization_{iteration}_design_space","x",s=30,facecolors='k',edgecolors='k')

def vis_objective_space(data,iteration):
    """
    Visualize the design space in objective coordinates.

    Arguments:
        res (pymoo.model.result.Result): Results object.
        iteration (int): Iteration number.
    """
    scatter_pymoo(data,f"optimization_{iteration}_objective_space_scatter","f",s=30,facecolors='w',edgecolors='k')

def vis_objective_space_pcp(data,iteration):
    """
    Visualize the design space in objective coordinates with the parallel coordinates plot.

    Arguments:
        data (np.array): Multidimensional Pareto front.
        iteration (int): Iteration number.
    """
    pcp(data,f"optimization_{iteration}_objective_space_pcp")

def compare_surrogate(inputs,outputs,predict,iteration):
    """
    Plot the comparison of raw data and surrogate response.

    Args:
        inputs (np.array): Input data.
        outputs (np.array): Output data.
        predict (method): Predict method of the surrogate.
        iteration (int): Iteration number.
    """
    data_all = np.stack((outputs.flatten(),predict(inputs).flatten()),1)
    scatter(data_all,f"iteration_{iteration}_surrogate",compare=True)

def sample_size_convergence(metrics):
    """
    Plot the sample size convergence.

    Arguments:
        metrics (dict): Dictionary of convergence metrics.
    """
    curve(metrics["values"],f"ssd_metric_{metrics['name']}",labels=["Iteration",metrics["name"].upper()],units=["-","-"])

def correlation_heatmap(predict,dim_in):
    """
    Plot the correleation heatmap between variables.

    Args:
        predict (method): Predict method of the surrogate.
        dim_in (int): Number of input dimensions.
    """    
    from datamod.sampling import sample ### Imported only here to avoid cyclical imports
    x = sample("grid",1000,dim_in)
    y = predict(x)
    data = np.concatenate((x,y),1)
    cor = np.corrcoef(data,rowvar=False)
    heatmap(cor)

def surrogate_response(inputs,outputs,iteration):
    """
    Plot the surrogate response.

    Args:
        inputs (np.array): Input data.
        outputs (np.array): Output to plot.
        iteration (int): Iteration number.
    """    
    data_all = np.concatenate((inputs,outputs),1)
    surface_pymoo(data_all,iteration)

def plot_training_history(history,train_in,train_out,test_in,test_out,predict,progress):
    """
    Plot the evolution of the training and testing error.

    Arguments:
        history (tensorflow.python.keras.callbacks.History/metamod.ANN_pt.TrainHistory): Metrics history during the training.
        train_in (np.array/torch.Tensor): Training input data.
        train_out (np.array/torch.Tensor): Training output data.
        test_in (np.array/torch.Tensor): Testing input data.
        test_out (np.array/torch.Tensor): Testing output data.
        predict (method): Predict method of the surrogate.
        progress (list): Training progress status.
    """
    learning_curves(history.history['loss'],history.history['val_loss'],train_out,predict(train_in),test_out,predict(test_in),progress)

def compare_pareto_fronts(pf_true,pf_calc):
    """
    Compare 2D Pareto fronts.

    Args:
        pf_true (np.array): True Pareto front.
        pf_calc (np.array): Calculated Pareto front.
    """    
    pareto_fronts(pf_true,pf_calc)

def plot_adaptive_candidates(candidates,data,iteration):
    """
    Plot candidates for adaptive sampling.

    Args:
        candidates (np.array): Candidate samples.
        data (np.array): Combined adaptive sampling metric.
        iteration (int): Iteration number.
    """    
    adaptive_candidates(candidates,data,iteration)
