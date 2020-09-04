"""
This is the visualization module.

visual - scatter 2D/3D, curve, surface tri (not quad)
"""

# Import pypi packages
import numpy as np

# Import custom packages
from visumod.plots import scatter,scatter_pymoo,curve,heatmap,pcp,surface_pymoo,learning_curves,pareto_fronts,adaptive_candidates

def plot_raw(data,iteration,normalized=False):
    """
    Plot either a scatter, curve or surface plot.

    Arguments:
        data: data object
        name: visualization type
        normalized: whether the data is normalized

    Raises:
        ValueError: if the visualization method is not supported

    Notes:
        * surface plot not used
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
        res: results object
        
    """
    scatter_pymoo(data,f"optimization_{iteration}_design_space","x",s=30,facecolors='k',edgecolors='k')

def vis_objective_space(data,iteration):
    """
    Visualize the design space in objective coordinates.

    Arguments:
        res: results object
        
    """
    scatter_pymoo(data,f"optimization_{iteration}_objective_space_scatter","f",s=30,facecolors='w',edgecolors='k')

def vis_objective_space_pcp(data,iteration):
    """
    Visualize the design space in objective coordinates.

    Arguments:
        res: results object
        
    """
    pcp(data,f"optimization_{iteration}_objective_space_pcp")

def compare_surrogate(inputs,outputs,predict,iteration):
    """
    Plot the comparison of raw data and surrogate response.
    """
    data_all = np.stack((outputs.flatten(),predict(inputs).flatten()),1)
    scatter(data_all,f"iteration_{iteration}_surrogate",compare=True)

def sample_size_convergence(metrics):
    """
    Plot the sample size convergence.

    Arguments:
        model: model object
        
    """
    curve(metrics["values"],f"ssd_metric_{metrics['name']}",labels=["Iteration",metrics["name"].upper()],units=["-","-"])

def correlation_heatmap(predict,dim_in):
    from datamod.sampling import sample
    x = sample("grid",1000,dim_in)
    y = predict(x)
    data = np.concatenate((x,y),1)
    cor = np.corrcoef(data,rowvar=False)
    heatmap(cor)

def surrogate_response(inputs,outputs,dimensions,iteration):
    data_all = np.concatenate((inputs,outputs),1)
    surface_pymoo(data_all,iteration)

def plot_training_history(history,train_in,train_out,test_in,test_out,predict,progress,trial_id=None):
    """
    Plot the evolution of the training and testing error.

    Arguments:
        history: training history object

    Raises:
        ValueError: if the surrogate has not been trained yet
        
    """
    learning_curves(history.history['loss'],history.history['val_loss'],train_out,predict(train_in),test_out,predict(test_in),progress,trial_id)

def compare_pareto_fronts(pf_true,pf_calc):
    pareto_fronts(pf_true,pf_calc)

def plot_adaptive_candidates(candidates,data,iteration):
    adaptive_candidates(candidates,data,iteration)
